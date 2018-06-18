/* A small header file for reading in CSV files stored as doubles, into an
 * array. */
#ifndef CSV_H_
#define CSV_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 128*1024

int getfield(char *line, double *row) {
    // The most important function used. It reads `line` from a csv into `row`
    // pointer. Honestly don't really understand it!
    const char* tok;
    int i = 0;
    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
        row[i] = atof(tok);  // atof turns a sting into a double
        i += 1;
    }

    return 0;
}


int getdim(FILE *stream, int ignore_headers, int *rows, int *cols) {
    // Return the row and columns of the csv file `stream`.

    // Keep getting new rows until we run out to get total rows. Also for the
    // first row, get the total columns by counting the commas.

    // Define variable for holding the information contained on each line.
    char line[BUFFER_SIZE];

    // If we ignore headers, get a discard the first row.
    if (ignore_headers == 1) {
        fgets(line, BUFFER_SIZE, stream);
    }

    // Loop until we can't get another line.
    *rows = 0;
    *cols = 0;
    while (fgets(line, BUFFER_SIZE, stream)) {
        if (*rows == 0) {
            // Get the column size.
            const char* tok;
            for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
                *cols += 1;
            }
        }
        *rows += 1;
    }

    return 0;
}



double *getcsv(const char *fname, int ignore_headers, int *rows, int *cols) {
    // Reads a CSV file containing multiple lines into a double array. As we
    // cannot know the size of the CSV beforehand, we define the array within
    // the function.

    FILE *stream;

    // Initially get the size of the csv.
    stream = fopen(fname, "r");
    getdim(stream, ignore_headers, rows, cols);

    // Point back to start of file.
    fseek(stream, 0, SEEK_SET);

    // Allocate memory for holding the csv.
    double *arr = malloc((*rows)*(*cols)*sizeof(double));

    // Define variable for holding the information contained on each line.
    char line[BUFFER_SIZE];

    // If we ignore headers, get a discard the first row.
    if (ignore_headers == 1) {
        fgets(line, BUFFER_SIZE, stream);
    }

    // Loop until we can't get another line.
    int i = 0;
    while (fgets(line, BUFFER_SIZE, stream)) {
        char* temp = strdup(line);
        getfield(temp, &arr[i*(*cols)]);
        // NOTE strtok clobbers temp hence why we duplicated.
        free(temp);
        i += 1;
    }

    fclose(stream);

    return arr;
}

int test_getfield() {
    // Test we can correctly parse a line from a CSV.
    printf("\ntest_getfield:\n");
    char line[] = "12, 21.2, 13.5,";
    char *temp = strdup(line);
    double true_row[] = {12, 21.2, 13.5};
    double row[3];
    int i;
    int test_passed = 1;

    getfield(temp, row);

    printf("True Output: ");
    printf("%s", line);

    printf("\nParsed Output: ");
    for (i = 0; i < 3; i++) {
        printf("%g, ", row[i]);
        if (row[i] != true_row[i]) {
            test_passed = -1;
        }
    }

    if (test_passed == 1) {
        printf("\nTest Passed!\n");
    } else {
        printf("\nTest Failed!\n");
    }
    return test_passed;
}

int test_getdim() {
    // Get dimensions of a basic csv file.
    printf("\ntest_getdim:\n");
    int test_passed = 1;
    int ignore_headers = 0;
    int rows;
    int cols;
    int true_rows = 3;
    int true_cols = 4;

    // Load the file.
    FILE *stream;
    stream = fopen("./data/example.csv", "r");

    getdim(stream, ignore_headers, &rows, &cols);
    fclose(stream);

    // Print.
    printf("True (Rows, Cols): ");
    printf("%d, %d", true_rows, true_cols);

    printf("\nParsed (Rows, Cols): ");
    printf("%d, %d", rows, cols);

    if ((true_rows == rows) & (true_cols == cols)) {
        test_passed = 1;
    } else {
        test_passed = -1;
    }

    // See if test passed.
    if (test_passed == 1) {
        printf("\nTest Passed!\n");
    } else {
        printf("\nTest Failed!\n");
    }
    return test_passed;
}

int test_getcsv() {
    // Get dimensions of a basic csv file.
    printf("\ntest_getcsv:\n");
    int test_passed = 1;
    int ignore_headers = 0;
    int rows;
    int cols;
    int true_rows = 3;
    int true_cols = 4;
    double true_values[3][4] = {
        {1, 2, 5, 6.6},
        {9, 3, 4.2, 1},
        {3, 2, 0.1, 2}
    };

    // Load the file.
    const char* fname = "./data/example.csv";

    double *matrix = getcsv(fname, ignore_headers, &rows, &cols);

    // Print dimensions.
    printf("True (Rows, Cols): ");
    printf("%d, %d", true_rows, true_cols);

    printf("\nParsed (Rows, Cols): ");
    printf("%d, %d", rows, cols);

    if ((true_rows == rows) & (true_cols == cols)) {
        test_passed = 1;
    } else {
        test_passed = -1;
    }

    // Print values.
    printf("\nTrue csv values:");
    int i, j;
    for (i = 0; i < rows; i++) {
        printf("\n");
        for (j = 0; j < cols; j++) {
            printf("%g, ", true_values[i][j]);
        }
    }

    printf("\nParsed csv values:");
    for (i = 0; i < rows; i++) {
        printf("\n");
        for (j = 0; j < cols; j++) {
            printf("%g, ", matrix[i*cols + j]);

            if (matrix[i*cols + j] != true_values[i][j]) {
                test_passed = -1;
            }
        }
    }

    // See if test passed.
    if (test_passed == 1) {
        printf("\nTest Passed!\n");
    } else {
        printf("\nTest Failed!\n");
    }

    free(matrix);

    return test_passed;
}

#endif
