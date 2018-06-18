#include <stdio.h>

// C program to implement interpolation search.
//     https://www.geeksforgeeks.org/interpolation-search
// If x is present in arr[0..n-1], then returns
// index of it, else returns -1.
int interpolationSearch(int arr[], int n, int x, int *depth) {
    // Find indexes of two corners
    int lo = 0, hi = (n - 1);

    // With the case where lo and hi are equal.
    // If x is equal to them, then all is good.
    if (lo == hi) {
        if (x == arr[lo])
            return 0;
        else
            return -1;
    }

    // Since array is sorted, an element present
    // in array must be in range defined by corner
    *depth = 0;
    while (lo <= hi && x >= arr[lo] && x <= arr[hi])
    {
        *depth += 1;

        // Probing the position with keeping
        // uniform distribution in mind.
        int pos;
        if (hi == lo) {
            // Avoid a division by 0 here.
            pos = lo;
        } else {
            pos = lo + (((double)(hi-lo) /
                  (arr[hi]-arr[lo]))*(x - arr[lo]));
        }

        // Condition of target found
        if (arr[pos] == x)
            return pos;

        // If x is larger, x is in upper part
        if (arr[pos] < x)
            lo = pos + 1;

        // If x is smaller, x is in the lower part
        else
            hi = pos - 1;
    }
    return -1;
}

// C program to implement iterative Binary Search
//     https://www.geeksforgeeks.org/binary-search/
// A iterative binary search function. It returns
// location of x in given array arr[l..r] if present,
// otherwise -1
int binarySearch(int arr[], int n, int x, int *depth) {
    int lo = 0;
    int hi = n-1;

    *depth = 0;
    while (lo <= hi)
    {
        *depth += 1;

        // Get middle index.
        int pos = lo + (hi-lo)/2;

        // Check if x is present at mid.
        if (arr[pos] == x)
            return pos;

        // If x greater, ignore left half.
        if (arr[pos] < x)
            lo = pos + 1;

        // If x is smaller, ignore right half.
        else
            hi = pos - 1;
    }

    // if we reach here, then element was
    // not present
    return -1;
}

/*
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

// Driver Code
int main() {
    // Array of items on which search will
    // be conducted.
    int arr[] =  {10, 100, 1000, 10000, 100000, 1000000, 10000000};
    int n = sizeof(arr)/sizeof(arr[0]);

    int x = 10000; // Element to be searched
    int depth;

    // Interpolation search.
    int index = interpolationSearch(arr, n, x, &depth);

    // If element was found
    printf("Interpolation Search:\n");
    if (index != -1)
        printf("\tElement found at index %d after depth %d.", index, depth);
    else
        printf("\tElement not found after depth %d.", depth);

    printf("\n");

    // Binary search.
    index = binarySearch(arr, n, x, &depth);

    // If element was found
    printf("Binary Search:\n");
    if (index != -1)
        printf("\tElement found at index %d after depth %d.", index, depth);
    else
        printf("\tElement not found after depth %d.", depth);

    printf("\n");

    return 0;
}
*/
