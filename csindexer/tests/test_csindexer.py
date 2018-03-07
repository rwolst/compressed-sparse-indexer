import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
from contexttimer import Timer

from csindexer import indexer

def test_get_small():
    matrix = [[ 0.  ,  0.  ,  0.45],
              [ 0.22,  0.74,  0.87],
              [ 0   ,  0   ,  0   ],
              [ 0   ,  0.6 ,  0   ],
              [ 0.  ,  0.93,  0.  ]]

    M = []
    M.append(sp.sparse.csr_matrix(matrix))
    M.append(sp.sparse.csc_matrix(matrix))

    row_vector = np.array([0, 0, 1, 1, 4, 4, 4], dtype=np.int32)
    col_vector = np.array([2, 2, 0, 1, 1, 1, 1], dtype=np.int32)
    indices = np.concatenate([row_vector[:, None], 
                              col_vector[:, None]], axis=1)
    true = np.array([0.45, 0.45, 0.22, 0.74, 0.93, 0.93, 0.93])
    for i in range(2):
        if i == 0:
            print('\nCSR matrix')
        else:
            print('\nCSC matrix')

        with Timer() as t:
            if i == 0:
                # Sort indices according to row first
                sort_idx = np.lexsort((indices[:,1], indices[:,0]))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((indices[:,0], indices[:,1]))
        print('\tLexsort time: %s' % t.elapsed)
        
        with Timer() as t:
            data_cy = np.empty(indices.shape[0], dtype=np.float64)
            indexer.apply(M[i], 
                          np.array(indices[sort_idx,0]), 
                          np.array(indices[sort_idx,1]),
                          data_cy,
                          'get')

            # Unsort data_cy
            unsort_idx = np.argsort(sort_idx)
            data_cy = data_cy[unsort_idx]

        print('\tCython time to get: %s' % t.elapsed)

        with Timer() as t:
            data_py = np.squeeze(np.array(M[i][indices[:,0], indices[:,1]]))
        print('\tPython time to get: %s' % t.elapsed)

        assert(np.all((data_cy - true)**2 < 1e-6))
        assert(np.all((data_py - true)**2 < 1e-6))

def test_add_small():
    matrix = [[ 0.  ,  0.  ,  0.45],
              [ 0.22,  0.74,  0.87],
              [ 0   ,  0   ,  0   ],
              [ 0   ,  0.6 ,  0   ],
              [ 0.  ,  0.93,  0.  ]]

    M = []
    M.append(sp.sparse.csr_matrix(matrix))
    M.append(sp.sparse.csc_matrix(matrix))

    row_vector  = np.array([0, 0, 1, 1, 4, 4, 4], dtype=np.int32)
    col_vector  = np.array([2, 2, 0, 1, 1, 1, 1], dtype=np.int32)
    data_vector = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    indices = np.concatenate([row_vector[:, None], 
                              col_vector[:, None]], axis=1)
    true_csr = np.array([2.45, 1.22, 1.74, 0.87, 0.6, 3.93])
    true_csc = np.array([1.22, 1.74, 0.6, 3.93, 2.45, 0.87])
    for i in range(2):
        M_copy = M[i].copy()
        if i == 0:
            print('\nCSR matrix')
        else:
            print('\nCSC matrix')

        with Timer() as t:
            if i == 0:
                # Sort indices according to row first
                sort_idx = np.lexsort((indices[:,1], indices[:,0]))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((indices[:,0], indices[:,1]))
        print('\tLexsort time: %s' % t.elapsed)
        
        with Timer() as t:
            indexer.apply(M[i], 
                          np.array(indices[sort_idx,0]), 
                          np.array(indices[sort_idx,1]),
                          data_vector[sort_idx],
                          'add')

        print('\tCython time to add: %s' % t.elapsed)

        with Timer() as t:
            #import pdb; pdb.set_trace()
            df = pd.DataFrame(np.concatenate([indices, data_vector[:,None]], axis=1), 
                    columns=['Row', 'Col', 'Data'])
            df = df.groupby(['Row', 'Col']).sum()
            df_row = df.index.get_level_values('Row').astype(np.int32)
            df_col = df.index.get_level_values('Col').astype(np.int32)
            df_data = df['Data']
            M_copy[df_row, df_col] = M_copy[df_row, df_col] + df_data.as_matrix()

        print('\tPython time to add: %s' % t.elapsed)

        if i == 0:
            assert(np.all((M[i].data - true_csr)**2 < 1e-6))
            assert(np.all((M_copy.data - true_csr)**2 < 1e-6))
        else:
            assert(np.all((M[i].data - true_csc)**2 < 1e-6))
            assert(np.all((M_copy.data - true_csc)**2 < 1e-6))

def test_get_large():
    matrix = sp.sparse.rand(40000, 40000, density=0.001)

    M = []
    M.append(sp.sparse.csr_matrix(matrix))
    M.append(sp.sparse.csc_matrix(matrix))

    # Indices must be a choice of the indices in matrix
    idx = np.random.choice(matrix.nnz, 1000000, replace=True)
    row_vector = matrix.row[idx]
    col_vector = matrix.col[idx]
    indices = np.concatenate([row_vector[:, None], 
                              col_vector[:, None]], axis=1)


    for i in range(2):
        if i == 0:
            print('\nCSR matrix')
        else:
            print('\nCSC matrix')

        with Timer() as t:
            if i == 0:
                # Sort indices according to row first
                sort_idx = np.lexsort((indices[:,1], indices[:,0]))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((indices[:,0], indices[:,1]))
        print('\tLexsort time: %s' % t.elapsed)
        
        with Timer() as t:
            data_cy = np.empty(indices.shape[0], dtype=np.float64)
            indexer.apply(M[i], 
                          np.array(indices[sort_idx,0]), 
                          np.array(indices[sort_idx,1]),
                          data_cy,
                          'get')

            # Unsort data_cy
            unsort_idx = np.argsort(sort_idx)
            data_cy = data_cy[unsort_idx]

        print('\tCython time to get: %s' % t.elapsed)

        with Timer() as t:
            data_py = np.squeeze(np.array(M[i][indices[:,0], indices[:,1]]))
        print('\tPython time to get: %s' % t.elapsed)


    assert(np.all((data_cy - data_py)**2 < 1e-6))

def test_add_large():
    matrix = sp.sparse.rand(40000, 40000, density=0.001)

    M = []
    M.append(sp.sparse.csr_matrix(matrix))
    M.append(sp.sparse.csc_matrix(matrix))

    # Indices must be a choice of the indices in matrix
    idx = np.random.choice(matrix.nnz, 1000000, replace=True)
    #import pdb; pdb.set_trace()
    row_vector = matrix.row[idx]
    col_vector = matrix.col[idx]
    data_vector = np.random.rand(idx.size).astype(np.float64)
    indices = np.concatenate([row_vector[:, None], 
                              col_vector[:, None]], axis=1)


    for i in range(2):
        M_copy_cy = M[i].copy()
        M_copy_py = M[i].copy()
        if i == 0:
            print('\nCSR matrix')
        else:
            print('\nCSC matrix')

        with Timer() as t:
            if i == 0:
                # Sort indices according to row first
                sort_idx = np.lexsort((indices[:,1], indices[:,0]))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((indices[:,0], indices[:,1]))
        print('\tLexsort time: %s' % t.elapsed)
        
        with Timer() as t:
            indexer.apply(M_copy_cy, 
                          np.array(indices[sort_idx,0]), 
                          np.array(indices[sort_idx,1]),
                          np.array(data_vector[sort_idx]),
                          'add')

        print('\tCython time to add: %s' % t.elapsed)

        with Timer() as t:
            #import pdb; pdb.set_trace()
            df = pd.DataFrame(np.concatenate([indices, data_vector[:,None]], axis=1), 
                    columns=['Row', 'Col', 'Data'])
            df = df.groupby(['Row', 'Col']).sum()
            df_row = df.index.get_level_values('Row').astype(np.int32)
            df_col = df.index.get_level_values('Col').astype(np.int32)
            df_data = df['Data']
            M_copy_py[df_row, df_col] = M_copy_py[df_row, df_col] + df_data.as_matrix()

        print('\tPython time to add: %s' % t.elapsed)

        if i == 0:
            assert(np.all((M_copy_cy.data - M_copy_py.data)**2 < 1e-6))
        else:
            assert(np.all((M_copy_cy.data - M_copy_py.data)**2 < 1e-6))
