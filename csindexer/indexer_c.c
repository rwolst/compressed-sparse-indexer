#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "indexer_c.h"
#include "interpolation_search.h"
#include "csv.h"

int get_first_occurence(int arr[], int n, int x, int *depth, int search_type) {
    // Use a binary or interpolation search to get the first occurence of a
    // value `x` in an array `arr` of size `n`. The search used can be either
    //     search_type:
    //         0: Binary search.
    //         1: Interpolation search.

    int idx;
    switch (search_type) {
        case 0:
            idx = binarySearch(arr, n, x, depth);
            break;

        case 1:
            idx = interpolationSearch(arr, n, x, depth);
            break;

        case 2:
            idx = jointSearch(arr, n, x, depth);
            break;
    }

    // Handle the case where the idx is not found.
    if (idx == -1) {
        printf("\nValue not found in array in get_first_occurence.\n");
        return -1;
    }

    // Now find the first occurence in the array.
    int i;
    if (idx == 0) {
        // First value must of course be 0.
        return 0;
    } else {
        // Search backwards through arr until we have a value not equal to
        // arr[idx].
        for (i=1; i<idx+1; i++) {
            if (arr[idx - i] != arr[idx]) {
                // We have found our first value!
                return idx - i + 1;
            }
        }

        // If we reach this point then again the first value is at 0.
        return 0;
    }
}

void process_row(int index_pointer, CS *M, COO *indexer, void (*f)(double *,
                 double *), int *axis0, int *axis1) {
    int sparse_pointer = M->indptr[axis0[index_pointer]];
    // printf("\n\t\t(Index pointer, Sparse pointer): (%d, %d)", index_pointer, sparse_pointer);

    int new_axis0 = 0;

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

void compressed_sparse_index_sorted(CS *M, COO *indexer,
                             void (*f)(double *, double *),
                             int n_threads) {
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

    // Loop over all values of our indexer.
    // Each of our threads should read the next row of a queue and then
    // process that row themselves.

    // Get where the rows start in indexer.
    // Unfortunately requires two passes over indexer, first find total rows
    // then find where they start.
    int i;
    int prev_row = -1;
    int total_rows = 0;
    for (i=0; i<indexer->nnz; i++) {
        if (axis0[i] != prev_row) {
            // We have a new row (or column).
            total_rows += 1;
            prev_row = axis0[i];
        }
    }
    // printf("\nTotal rows: %d", total_rows);
    // printf("\n");
    int *row_start = malloc(total_rows*sizeof(int));
    prev_row = -1;
    total_rows = 0;
    for (i=0; i<indexer->nnz; i++) {
        if (axis0[i] != prev_row) {
            // We have a new row (or column).
            row_start[total_rows] = i;
            total_rows += 1;
            prev_row = axis0[i];
        }
    }


    // Can parallelise the below for loop.
    // printf("\n\tNew indexing");
    if (n_threads != -1) {
        omp_set_num_threads(n_threads);
    }
    #pragma omp parallel for
    for (i=0; i<total_rows; i++) {
        process_row(row_start[i], M, indexer, f, axis0, axis1);
    }


    free(row_start);
}



void compressed_sparse_index(CS *M, COO *indexer,
                             void (*f)(double *, double *), int search_type,
                             int n_threads) {
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
    // Only worth openMP for complex f value otherwise just makes it slower.
    if (n_threads != -1) {
        omp_set_num_threads(n_threads);
    }

    #pragma omp parallel for schedule(dynamic) shared(M, indexer)
    for (index_pointer=0; index_pointer<indexer->nnz; index_pointer++) {
        // If we can guarantee all values in indexer exist in M then we can
        // use our binary search for the current column value
        //     axis1[index_pointer].
        int idx;
        int depth;
        int start;
        int n;
        int x;

        start = M->indptr[axis0[index_pointer]];
        n = M->indptr[axis0[index_pointer]+1] - start;
        x = axis1[index_pointer];
        idx = get_first_occurence(&M->indices[start], n, x, &depth, search_type);

        // Now apply our function at the correct index.
        (*f)(&(M->data[start+idx]), &(indexer->data[index_pointer]));
        // indexer->data[index_pointer] = M->data[start+idx];
    }
}

void get(double *x, double *y) {
    // Copy x value into y
    double temp;
    #pragma omp atomic read
    temp = *x;

    #pragma omp atomic write
    *y = temp;
}

void set(double *x, double *y) {
    // Copy y value into x
    // Note that sometimes the same entry gets multiple copies
    double temp;
    #pragma omp atomic read
    temp = *y;

    #pragma omp atomic write
    *x = temp;
}

void add(double *x, double *y) {
    // Add x value into y
    #pragma omp atomic
    *x += *y;
}

int example_get() {
    // A small example to check we can get from a CS matrix
    int i;
    int search_type = 0;
    int n_threads = 1;

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

    compressed_sparse_index_sorted(&M, &indexer, get, n_threads);
    compressed_sparse_index(&M, &indexer, get, search_type, n_threads);

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
    int search_type = 0;
    int n_threads = 1;

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

    compressed_sparse_index_sorted(&M, &indexer, add, n_threads);
    compressed_sparse_index(&M, &indexer, add, search_type, n_threads);

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


int python_debugger() {
    // A function for testing the C code especially when we get seg faults in
    // Python.

    // Load the CSV files and store in correct objects.
    int i, rows, cols;
    char fname[128];
    int n_threads = 1;

    // Index object.
    COO indexer;

    //     row_vec
    strcpy(fname, "tests/data/row_vec.csv");
    double *arr = getcsv(fname, 0, &rows, &cols);

    // Convert to integers.
    int *arr_int = malloc(rows*sizeof(int));
    for (i = 0; i < rows; i++) {
        arr_int[i] = (int) arr[i];
    }
    free(arr);

    indexer.row = arr_int;

    //     col_vec
    strcpy(fname, "tests/data/col_vec.csv");
    arr = getcsv(fname, 0, &rows, &cols);

    // Convert to integers.
    arr_int = malloc(rows*sizeof(int));
    for (i = 0; i < rows; i++) {
        arr_int[i] = (int) arr[i];
    }
    free(arr);

    indexer.col = arr_int;

    //     data
    strcpy(fname, "tests/data/col_vec.csv");
    arr = getcsv(fname, 0, &rows, &cols);
    indexer.data = arr;

    indexer.nnz = rows;

    // CSR object.
    CS M;
    M.CSR = 1;

    //     indptr
    strcpy(fname, "tests/data/indptr.csv");
    arr = getcsv(fname, 0, &rows, &cols);

    // Convert to integers.
    arr_int = malloc(rows*sizeof(int));
    for (i = 0; i < rows; i++) {
        arr_int[i] = (int) arr[i];
    }
    free(arr);

    M.indptr = arr_int;
    M.n_indptr = rows;

    //     indices
    strcpy(fname, "tests/data/indices.csv");
    arr = getcsv(fname, 0, &rows, &cols);

    // Convert to integers.
    arr_int = malloc(rows*sizeof(int));
    for (i = 0; i < rows; i++) {
        arr_int[i] = (int) arr[i];
    }
    free(arr);

    M.indices = arr_int;

    //     data
    strcpy(fname, "tests/data/data.csv");
    arr = getcsv(fname, 0, &rows, &cols);

    M.data = arr;

    // Run the program.
    compressed_sparse_index(&M, &indexer, get, 2, n_threads);

    // Free indexer.
    free(indexer.row);
    free(indexer.col);
    free(indexer.data);

    // Free CSR.
    free(M.indptr);
    free(M.indices);
    free(M.data);
}

int main() {
    python_debugger();
    //example_get();
    //example_add();
    //example_split_perfect();
    //example_split_ugly();
    //example_split_ugly2();
    //example_split_multiple();
    //example_split_equal();

    return 0;
}
