#include <stdlib.h>
#include <stdio.h>
#include "indexer_c.h"

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
        sparse_pointer = M->indptr[axis0[index_pointer]];
        // printf("\n\t\t(Index pointer, Sparse pointer): (%d, %d)", index_pointer, sparse_pointer);

        new_axis0 = 0;

        // Now choose between incrementing index_pointer and the sparse_pointer based on what values
        // we get.
        while ((new_axis0 == 0) & 
               (sparse_pointer < M->indptr[axis0[index_pointer] + 1])) {
            /* While both the indexer and M are on the same axis
               We begin by pointing at the top of this axis of
               our vectors and gradually move down them. In the event of 
               an equality we apply our function and 
               increment the INDEXING VECTOR pointer, not the sparse
               vector pointer, as there can be multiple values that 
               are the same in the indexing vector but not the sparse row
               column vector (only 1 column can appear in 1 row!). */

            if (M->indices[sparse_pointer] == axis1[index_pointer]) {
                // Apply the function to their data
                (*f)(&(M->data[sparse_pointer]), &(indexer->data[index_pointer]));

                // Only increment the index pointer
                index_pointer += 1;
                
                // Check for a new axis in the COO indexer
                if (index_pointer >= indexer->nnz) {
                    break;
                }
                if (axis0[index_pointer] != axis0[index_pointer-1]) {
                    new_axis0 = 1;
                }
            } else if (M->indices[sparse_pointer] > axis1[index_pointer]) {
                // Need to increment index pointer
                index_pointer += 1;
                
                // Check for a new axis in the COO indexer
                if (index_pointer >= indexer->nnz) {
                    break;
                }
                if (axis0[index_pointer] != axis0[index_pointer-1]) {
                    new_axis0 = 1;
                }
            } else {
                // Need to increment sparse pointer
                sparse_pointer += 1;
            }

        }
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

int main() {
    // example_get();
    example_add();

    return 0;
}
