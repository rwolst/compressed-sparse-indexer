typedef struct {
    // A compressed sparse matrix (can be CSR or CSC)
    int CSR;  // Whether sparse matrix is CSR (otherwise we assume it is CSC)
    int *indptr;
    int *indices;
    double *data;
    int n_indptr;  // Length of indptr vector
} CS;

typedef struct {
    // A sparse matrix in COO format
    int *row;
    int *col;
    double *data;
    int nnz;
} COO;

void compressed_sparse_index(CS *M, COO *indexer, void (*f)(double *, double *));
 
void get(double *x, double *y);
void add(double *x, double *y);
