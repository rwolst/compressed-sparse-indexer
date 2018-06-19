import numpy as np
cimport numpy as np
from contexttimer import Timer

cdef extern from 'indexer_c.h':
    ctypedef struct CS:
        int CSR
        int *indptr
        int *indices
        double *data
        int n_indptr

    ctypedef struct COO:
        int *row
        int *col
        double *data
        int nnz

    void compressed_sparse_index(CS *M, COO *indexer, 
                                 void (*f)(double *, double *),
                                 int search_type, int n_threads);

    void get(double *x, double *y);
    void add(double *x, double *y);

def apply(M, 
          np.int32_t[:] row_vector, 
          np.int32_t[:] col_vector,
          np.float64_t[:] data_vector,
          operation,
          search_type,
          n_threads):
    """Gets M[row_vector, col_vector].
    If M is a CSR matrix, then 
        indices = [row_vector, col_vector]
    must be ordered by precedence (row, column). Technically the 
    rows can be in any order as long as blocks of rows are 
    together i.e.
        (1,0)
        (1,1)
        (1,2)
        (0,1)
        (0,2)
    would be ok.
    If M is a CSC matrix, then indices must be ordered by
    precedence (column, row).
    
    The variable search_type can be
        binary: Binary search.
        interpolation: Interpolation search.
        """
    cdef np.int32_t N = row_vector.size
    cdef CS M_CS
    cdef np.int32_t[:] indptr  = M.indptr
    cdef np.int32_t[:] indices = M.indices
    cdef np.float64_t[:] data  = M.data
    cdef COO indexer
    cdef np.int32_t search_type_int

    with Timer() as t:
        assert(row_vector.size == col_vector.size)
        assert(row_vector.size == data_vector.size)

        if search_type == 'binary':
            search_type_int = 0
        elif search_type == 'interpolation':
            search_type_int = 1
        elif search_type == 'joint':
            search_type_int = 2
        else:
            raise Exception("Unrecognised search_type: %s" % search_type)


        # Build the CS and COO structures
        if M.getformat() == 'csr':
            M_CS.CSR = 1
            M_CS.n_indptr = M.shape[0] + 1
        elif M.getformat() == 'csc':
            M_CS.CSR = 0
            M_CS.n_indptr = M.shape[1] + 1
        else:
            raise Exception('Sparse format %s not csr or csc' % M.getformat())

        # Get a C view on Python objects so we can take address
        M_CS.indptr  = <int *> &(indptr[0])
        M_CS.indices = <int *> &(indices[0])
        M_CS.data    = <double *> &(data[0])

        indexer.row = <int *> &(row_vector[0]) 
        indexer.col = <int *> &(col_vector[0]) 
        indexer.data = <double *> &(data_vector[0])
        indexer.nnz = N

        # Run our function with the get or add method
        if operation == 'get':
            compressed_sparse_index(&M_CS, &indexer, get, search_type_int,
                                    n_threads)
        if operation == 'add':
            compressed_sparse_index(&M_CS, &indexer, add, search_type_int,
                                    n_threads)
    print("\tCython internal time: %s" % t.elapsed)
