#include <stdlib.h>
#include <stdio.h>
#include "indexer_c.h"


int weighted_binary_split(int *x, int start_limit, int start_idx, int end_idx,
                          int val, int *depth) {
    // For a vector of length `n` with known starting and ending values,
    //     x[start_idx:end_idx+1] = [start, ..., end]
    // searches for `val` in `x`, assuming `x` is uniformly distributed and
    // ordered. Note that we always get the first occurence of `val` in `x`.
    // I say 'best' because it is simply intuition.
    // Note that when initially calling this, we will have
    //     start_limit = start_idx
    // however when we are within the recursion this is no longer the case.
    int n = end_idx - start_idx + 1;
    int start = x[start_idx];
    int end = x[end_idx];
    int out;
    int i;

    // Keep track of how many iterations/depth we have gone in binary search
    // tree.
    *depth += 1;

    // Catch the case where end == start (causing divide by 0 below).
    if (end - start == 0) {
        return 0;
    }

    // Catch the case where val == start (can just return start_idx).
    if (val == start) {
        return start_idx;
    }

    // We want to find
    //     (val - start)/((end - start)/n) = n*val/(end - start),
    // rounded to the nearest integer.
    // Note there is no point using
    //     1) idx = start_idx 
    //        as if val == x[start_idx] ==> val == start, caught above.
    //     2) idx = start_idx + n = end_idx + 1, too big for vector.
    // Hence we use (n - 1).
    int idx = start_idx + ((n-1)*(val - start))/(end - start);
    if (idx == start_idx) {
        idx += 1;
    }

    if (x[idx] > val) {
        // Search again with idx as the new `end_idx`.
        out = weighted_binary_split(x, start_limit, start_idx, idx, val, depth);
    } else if (x[idx] < val) {
        // Search again with idx as the new `start_idx`.
        out = weighted_binary_split(x, start_limit, idx, end_idx, val, depth);
    } else {
        // In this case we have found val in x at x[idx].
        // We finally get the first occurence of `val` in x[start_limit:idx+1].
        for (i=1; i<idx-start_limit+1; i++) {
            if (x[idx - i] != x[idx]) {
                // We have found our first value!
                return idx - i + 1;
            }
        }

        // If we reach this point, then x[start_limit] was the first value.
        return start_limit;
    }

    return out;
}


void compressed_sparse_index(CS *M, COO *indexer, void (*f)(double *, double *)) {
    /*
    Note we can maybe split the indexer into separate chunks and
    perform our operations in parallel over the chunks.
    Inputs:
        M: A compressed sparse matrix in CSC or CSR form to get/set etc.
        index: A sparse matrix in COO form containing index into M.
               If M is CSC it is assumed to be ordered by (row, column) and
               if M is CSR it is assumed to be ordered by (column, row).
        f: A function taking f(&M->data[x], &index->data[y]) and applies something to them.
    */
    int new_axis0;
    int sparse_pointer;
    int index_pointer = 0;
    int idx;
    int depth = 0;

    int *axis0;
    int *axis1;

    // Create view onto rows/cols of the COO matrx to make updates independent of
    // whether M is stored as a CSC or CSR.
    if (M->CSR == 1) {
        axis0 = indexer->row;
        axis1 = indexer->col;
    } else {
        axis1 = indexer->row;
        axis0 = indexer->col;
    }

    // Loop over all values of our indexer
    // printf("\n\tNew indexing");
    while (index_pointer < indexer->nnz) {
        // If we are here we must have a new row
        // Sparse pointer points to columns
        // sparse_pointer = M->indptr[axis0[index_pointer]];
        // printf("\n\t\t(Index pointer, Sparse pointer): (%d, %d)", index_pointer, sparse_pointer);

        // If we can guarantee all values in indexer exist in M then we can
        // use our binary search for the current column value
        //     axis1[index_pointer].
        idx = weighted_binary_split(M->indices,
                                    M->indptr[axis0[index_pointer]],
                                    M->indptr[axis0[index_pointer]],
                                    M->indptr[axis0[index_pointer]+1]-1,
                                    axis1[index_pointer],
                                    &depth) +
              M->indptr[axis0[index_pointer]];

        // Now apply our function at the correct index.
        (*f)(&(M->data[idx]), &(indexer->data[index_pointer]));
        index_pointer += 1;
    }
}

void get(double *x, double *y) {
    // Copy x value into y
    *y = *x;
}

void set(double *x, double *y) {
    // Copy y value into x
    // Note that sometimes the same entry gets multiple copies
    *x = *y;
}

void add(double *x, double *y) {
    // Add x value into y
    *x = *x + *y;
}

int example_get() {
    // A small example to check we can get from a CS matrix
    int i;
    
    // Use M = [[ 0.  ,  0.  ,  0.45],
    //          [ 0.22,  0.74,  0.87],
    //          [ 0   ,  0   ,  0   ]
    //          [ 0   ,  0.6 ,  0   ],
    //          [ 0.  ,  0.93,  0.  ]]
    CS M;
    M.CSR = 1;
    M.n_indptr = 6;

    M.indptr  = malloc(6*sizeof(int));
    M.indices = malloc(6*sizeof(int));
    M.data    = malloc(6*sizeof(double));

    M.indptr[0] = 0; M.indptr[1] = 1; 
    M.indptr[2] = 4; M.indptr[3] = 4;
    M.indptr[4] = 5; M.indptr[5] = 6;

    M.indices[0] = 2; M.indices[1] = 0; 
    M.indices[2] = 1; M.indices[3] = 2;
    M.indices[4] = 1; M.indices[5] = 1;

    M.data[0] = 0.45; M.data[1] = 0.22; 
    M.data[2] = 0.74; M.data[3] = 0.87;
    M.data[4] = 0.60; M.data[5] = 0.93;

    // Use indexer = [[0, 2, _],
    //                [0, 2, _],
    //                [1, 0, _],
    //                [1, 1, _],
    //                [4, 1, _],
    //                [4, 1, _]
    //                [4, 1, _]]
    //
    // Note it is ordered by (row, column) and _ represents
    // the fact the data doesn't need initialising as its a 
    // get operation we are performing.
    COO indexer;
    indexer.nnz = 7;

    indexer.row = malloc(7*sizeof(int));
    indexer.col = malloc(7*sizeof(int));
    indexer.data = malloc(7*sizeof(double));

    indexer.row[0] = 0; indexer.col[0] = 2;
    indexer.row[1] = 0; indexer.col[1] = 2;
    indexer.row[2] = 1; indexer.col[2] = 0;
    indexer.row[3] = 1; indexer.col[3] = 1;
    indexer.row[4] = 4; indexer.col[4] = 1;
    indexer.row[5] = 4; indexer.col[5] = 1;
    indexer.row[6] = 4; indexer.col[6] = 1;

    compressed_sparse_index(&M, &indexer, get);

    for (i=0; i<7; i++) {
        printf("\nindexer.data[%d] = %g", i, indexer.data[i]);
    }
    printf("\n");

    free(indexer.row);
    free(indexer.col);
    free(indexer.data);
    free(M.indptr);
    free(M.indices);
    free(M.data);

}

int example_add() {
    // A small example to check we can get from a CS matrix
    int i;

    // Use M = [[ 0.1  ,  0.2  ,  0.3],
    //          [ 0.4  ,  0.5  ,  0.6],
    //          [ 0.7  ,  0.8  ,  0.9]]
    CS M;
    M.CSR = 1;
    M.n_indptr = 4;

    M.indptr  = malloc(4*sizeof(int));
    M.indices = malloc(9*sizeof(int));
    M.data    = malloc(9*sizeof(double));

    M.indptr[0] = 0; M.indptr[1] = 3; 
    M.indptr[2] = 6; M.indptr[3] = 9;

    M.indices[0] = 0; M.indices[1] = 1; M.indices[2] = 2; 
    M.indices[3] = 0; M.indices[4] = 1; M.indices[5] = 2; 
    M.indices[6] = 0; M.indices[7] = 1; M.indices[8] = 2; 

    M.data[0] = 0.1; M.data[1] = 0.2; M.data[2] = 0.3; 
    M.data[3] = 0.4; M.data[4] = 0.5; M.data[5] = 0.6; 
    M.data[6] = 0.7; M.data[7] = 0.8; M.data[8] = 0.9; 

    // Use indexer = [[1, 2, 0.5],
    //                [2, 2, 1.5]]
    //
    // Note it is ordered by (row, column) and _ represents
    // the fact the data doesn't need initialising as its a 
    // get operation we are performing.
    COO indexer;
    indexer.nnz = 2;

    indexer.row = malloc(indexer.nnz*sizeof(int));
    indexer.col = malloc(indexer.nnz*sizeof(int));
    indexer.data = malloc(indexer.nnz*sizeof(double));

    indexer.row[0] = 1; indexer.col[0] = 2; indexer.data[0] = 0.5;
    indexer.row[1] = 2; indexer.col[1] = 2; indexer.data[1] = 1.5;

    compressed_sparse_index(&M, &indexer, add);

    for (i=0; i<9; i++) {
        printf("\nM.data[%d] = %g", i, M.data[i]);
    }
    printf("\n");

    free(indexer.row);
    free(indexer.col);
    free(indexer.data);
    free(M.indptr);
    free(M.indices);
    free(M.data);

}


int example_split_perfect() {
    // Test that our weigted binary split works correctly for perfectly uniform
    // vector x (i.e. should only ever need depth == 1).
    int x[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int start_idx = 0;
    int end_idx = 9;
    int val = 3;
    int idx;
    int i;
    int depth = 0;

    idx = weighted_binary_split(x, start_idx, start_idx, end_idx, val, &depth);
    printf("\nFirst index into:\n\tx = (");
    for (i=0; i<10; i++) {
        printf("%d,", x[i]);
    }
    printf(")");
    printf("\nfor:\n\tval = %d", val);
    printf("\nis\n\tidx = %d", idx);
    printf("\nwith\n\tx[idx]=%d", x[idx]);
    printf("\nat\n\tdepth=%d\n", depth);

    return 0;
}


int example_split_ugly() {
    // Test that our weigted binary split works correctly for far from uniform
    // vector x (i.e. depth should be near linear).
    int x[10] = {0, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009};
    int start_idx = 0;
    int end_idx = 9;
    int val = 10001;
    int idx;
    int i;
    int depth = 0;

    idx = weighted_binary_split(x, start_idx, start_idx, end_idx, val, &depth);
    printf("\nFirst index into:\n\tx = (");
    for (i=0; i<10; i++) {
        printf("%d,", x[i]);
    }
    printf(")");
    printf("\nfor:\n\tval = %d", val);
    printf("\nis\n\tidx = %d", idx);
    printf("\nwith\n\tx[idx]=%d", x[idx]);
    printf("\nat\n\tdepth=%d\n", depth);

    return 0;
}

int example_split_ugly2() {
    // Test that our weigted binary split works correctly for far from uniform
    // vector x (i.e. depth should be near linear).
    int x[10] = {1, 2, 2, 2, 2, 2, 2, 2, 2, 1000};
    int start_idx = 0;
    int end_idx = 9;
    int val = 1;
    int idx;
    int i;
    int depth = 0;

    idx = weighted_binary_split(x, start_idx, start_idx, end_idx, val, &depth);
    printf("\nFirst index into:\n\tx = (");
    for (i=0; i<10; i++) {
        printf("%d,", x[i]);
    }
    printf(")");
    printf("\nfor:\n\tval = %d", val);
    printf("\nis\n\tidx = %d", idx);
    printf("\nwith\n\tx[idx]=%d", x[idx]);
    printf("\nat\n\tdepth=%d\n", depth);

    return 0;
}

int example_split_multiple() {
    // Test that our weigted binary split works correctly for multiple values
    // in vector x (i.e. finds the first value).
    int x[10] = {0, 2, 2, 4, 4, 5, 5, 5, 6, 6};
    int start_idx = 0;
    int end_idx = 9;
    int val = 6;
    int idx;
    int i;
    int depth = 0;

    idx = weighted_binary_split(x, start_idx, start_idx, end_idx, val, &depth);
    printf("\nFirst index into:\n\tx = (");
    for (i=0; i<10; i++) {
        printf("%d,", x[i]);
    }
    printf(")");
    printf("\nfor:\n\tval = %d", val);
    printf("\nis\n\tidx = %d", idx);
    printf("\nwith\n\tx[idx]=%d", x[idx]);
    printf("\nat\n\tdepth=%d\n", depth);

    return 0;
}

int example_split_equal() {
    // Test that our weigted binary split works correctly for equal values
    // in vector x (i.e. finds the first value).
    int x[10] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    int start_idx = 0;
    int end_idx = 9;
    int val = 2;
    int idx;
    int i;
    int depth = 0;

    idx = weighted_binary_split(x, start_idx, start_idx, end_idx, val, &depth);
    printf("\nFirst index into:\n\tx = (");
    for (i=0; i<10; i++) {
        printf("%d,", x[i]);
    }
    printf(")");
    printf("\nfor:\n\tval = %d", val);
    printf("\nis\n\tidx = %d", idx);
    printf("\nwith\n\tx[idx]=%d", x[idx]);
    printf("\nat\n\tdepth=%d\n", depth);

    return 0;
}

int main() {
    example_get();
    // example_add();
    //example_split_perfect();
    //example_split_ugly();
    //example_split_ugly2();
    //example_split_multiple();
    //example_split_equal();

    return 0;
}
