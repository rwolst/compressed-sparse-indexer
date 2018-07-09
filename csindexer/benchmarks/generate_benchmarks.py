"""This script generates benchmarks for the compressed sparse indexer versus
the scipy indexer. We note that we have the following parameters that can
change:
    1) SORT: Whether the indexing rows and columns are sorted.
    2) N_THREADS: Total threads to use when indexing.
    3) SPARSE_FORMAT: Use CSR or CSC format.
    4) ROWS: Total rows of sparse matrix to index.
    5) COLS: Total columns of sparse matrix to index.
    6) NNZ: Total non-zero values in sparse matrix to index.
    7) N_INDEXERS: Total number of points to index in sparse matrix.
    8) SEARCH_TYPE: Whether to use an internal
                        - binary search
                        - interpolation search
                        - joint search
    9) OPERATION: What function to apply on the indexed points:
                      - get
                      - set

Clearly this is a large space for our parameters to benchmark for and it is
hard to clearly investigate all their interacations.

Hence we perform all tests on both sorted and unsorted, CSR and CSC, get and
set and each search type. However for the other parameters we choose some
informative subsets for each test.
"""

import numpy as np
import scipy as sp
import scipy.sparse
from contexttimer import Timer
import matplotlib.pyplot as plt

from csindexer import indexer as csindexer

DEBUG = True


def index_time(sort, n_threads, sparse_format, rows, cols, nnz, n_indexers,
               search_type, operation):
    """A function for timing our cxindexer and scipy indexer. It first creates
    sparse matrices, sorts if necessary, runs indexers on both and returns
    the times."""
    if DEBUG:
        print("Benchmarking:\n\tSORT = %s\n\tN_THREADS = %s\n\tSPARSE_FORMAT ="
              " %s\n\tROWS = %s\n\tCOLS = %s\n\tNNZ = %s\n\tN_INDEXERS ="
              " %s\n\t" "SEARCH_TYPE = %s\n\tOPERATION = %s"
              % (sort, n_threads, sparse_format, rows, cols, nnz, n_indexers,
                 search_type, operation))

    # Generate matrix.
    with Timer() as t:
        M = sp.sparse.rand(rows, cols, density=nnz/(rows*cols))

    if DEBUG:
        print("\tTime to generate sparse matrix: %s" % t.elapsed)

    # Generate indexer.
    with Timer() as t:
        indexer = {}
        idx = np.random.choice(M.nnz, n_indexers, replace=True)
        indexer['row'] = M.row[idx]
        indexer['col'] = M.col[idx]
        indexer['data'] = np.random.rand(idx.size).astype(np.float64)

    if DEBUG:
        print("\tTime to generate indexer: %s" % t.elapsed)

    # Convert sparse matrix.
    with Timer() as t:
        if sparse_format == 'CSR':
            M = sp.sparse.csr_matrix(M)
        elif sparse_format == 'CSC':
            M = sp.sparse.csc_matrix(M)
        else:
            raise Exception("sparse_format must be either CSR or CSC.")

    if DEBUG:
        print("\tTime to convert sparse matrix: %s" % t.elapsed)

    # Sort.
    with Timer() as t:
        if sort:
            if sparse_format == 'CSR':
                # Sort indices according to row first
                sort_idx = np.lexsort((indexer['col'], indexer['row']))
            elif sparse_format == 'CSC':
                # Sort indices according to col first
                sort_idx = np.lexsort((indexer['row'], indexer['col']))
        else:
            sort_idx = np.arange(indexer['row'].size)

        unsort_idx = np.argsort(sort_idx)

    if DEBUG:
        print("\tTime to sort indexer: %s" % t.elapsed)
    sort_time = t.elapsed

    # Time the csindexer.
    with Timer() as t:
        ## Run the Cython function.
        if operation == 'get':
            ### Don't need to copy M as it doesn't get modified but do have
            ### to copy indexer['data'] as it does.
            data_cs = indexer['data'].copy()
            M_cs = M

            csindexer.apply(M_cs, np.array(indexer['row'][sort_idx]),
                            np.array(indexer['col'][sort_idx]), data_cs,
                            operation, search_type, n_threads, DEBUG)

            ### Unsort to get final result.
            data_cs = data_cs[unsort_idx]

        elif operation == 'add':
            ### Copy M, don't copy indexer['data'].
            data_cs = indexer['data']
            M_cs = M.copy()
            csindexer.apply(M_cs, np.array(indexer['row'][sort_idx]),
                            np.array(indexer['col'][sort_idx]),
                            np.array(data_cs[sort_idx]), operation,
                            search_type,
                            n_threads, DEBUG)
        else:
            raise Exception("Operation must be either get or add.")


    if DEBUG:
        print("\tTime for csindexer: %s" % t.elapsed)
    cs_time = t.elapsed

    # Time the scipy indexer.
    with Timer() as t:
        if operation == 'get':
            data_py = np.squeeze(np.array(M[indexer['row'][sort_idx],
                                            indexer['col'][sort_idx]]))
            data_py = data_py[unsort_idx]
        elif operation == 'add':
            M_sp = M.copy()

            idx_coo = sp.sparse.coo_matrix(
                (indexer['data'][sort_idx],
                 (indexer['row'][sort_idx], indexer['col'][sort_idx])),
                shape=(rows, cols))

            M_sp += idx_coo

    if DEBUG:
        print("\tTime for scipy indexer: %s" % t.elapsed)
    sp_time = t.elapsed

    # Check that both output same result.
    if operation == 'get':
        assert(np.all((data_cs - data_py)**2 < 1e-6))
    elif operation == 'add':
        assert(np.all((M_cs.data - M_sp.data)**2 < 1e-6))
        assert(np.all((M_cs.indptr - M_sp.indptr)**2 < 1e-6))
        assert(np.all((M_cs.indices - M_sp.indices)**2 < 1e-6))

    return cs_time, sp_time, sort_time


def plot_times(plt_info, ignore_params):
    """Function for plotting scipy and csindexer times."""
    # Set names of parameters we use.
    all_params = ['SORT', 'N_THREADS', 'SPARSE_FORMAT', 'ROWS', 'COLS',
                  'NNZ', 'N_INDEXERS', 'SEARCH_TYPE', 'OPERATION']

    # Get the maximum time seen.
    max_time = max([max([i['cs_times'].max(),
                         i['sp_times'].max(),
                         i['sort_times'].max()])
                    for i in plt_info])
    for info in plt_info:
        plt.figure(figsize=(20,20))
        plt.ylim(0, max_time)
        plt.plot(info['x_axis'], info['cs_times'], label='CS')
        plt.plot(info['x_axis'], info['sp_times'], label='SP')
        plt.plot(info['x_axis'], info['sort_times'], label='sort')
        plt.xlabel(info['name'])
        plt.legend()

        used_params = [i for i in all_params if i not in ignore_params]
        title = ""
        for idx, param in enumerate(used_params):
            if idx != 0:
                if idx % 2 == 0:
                    title += "\n"

            title += "%s: " % param
            title += "%s, " % info[param]
        plt.title(title)
        plt.show()


def test1():
    ## Test 1: How time changes on single thread, ROWS = COLS, with fixed NNZ
    ##         and N_INDEXERS.
    N_THREADS = 1
    NNZ = 1000
    N_INDEXERS = 10000
    x_axis = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]

    plt_info = []
    for SORT in [True, False]:
        for SPARSE_FORMAT in ['CSR', 'CSC']:
            for OPERATION in ['get', 'add']:
                for SEARCH_TYPE in ['binary', 'interpolation', 'joint']:
                    # Create arrays for holding timing values for csindexer and
                    # for scipy indexer.
                    cs_times = np.empty(len(x_axis))
                    sp_times = np.empty(len(x_axis))
                    sort_times = np.empty(len(x_axis))
                    for i, dim in enumerate(x_axis):
                        ROWS = dim
                        COLS = dim

                        out = index_time(SORT, N_THREADS,
                                                      SPARSE_FORMAT, ROWS,
                                                      COLS, NNZ, N_INDEXERS,
                                                      SEARCH_TYPE,
                                                      OPERATION)
                        cs_time, sp_time, sort_time = out

                        cs_times[i] = cs_time
                        sp_times[i] = sp_time
                        sort_times[i] = sort_time

                    # Save the info for plotting later.
                    plt_info.append({'x_axis': x_axis,
                                     'cs_times': cs_times,
                                     'sp_times': sp_times,
                                     'sort_times': sort_times,
                                     'SORT': SORT,
                                     'SPARSE_FORMAT': SPARSE_FORMAT,
                                     'OPERATION': OPERATION,
                                     'SEARCH_TYPE': SEARCH_TYPE,
                                     'ROWS': ROWS,
                                     'COLS': COLS,
                                     'N_THREADS': N_THREADS,
                                     'NNZ': NNZ,
                                     'N_INDEXERS': N_INDEXERS,
                                     'name': 'ROW & COL'})


    # Plot the times.
    plot_times(plt_info, ['ROWS', 'COLS'])

def test2():
    ## Test 2: How time changes on single thread, ROWS = COLS = 10000, with
    ##         fixed N_INDEXERS and varying NNZ.
    N_THREADS = 1
    ROWS = 10000
    COLS = 10000
    N_INDEXERS = 100000
    x_axis = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 1000000]

    plt_info = []
    for SORT in [True, False]:
        for SPARSE_FORMAT in ['CSR', 'CSC']:
            for OPERATION in ['get', 'add']:
                for SEARCH_TYPE in ['binary', 'interpolation', 'joint']:
                    # Create arrays for holding timing values for csindexer and
                    # for scipy indexer.
                    cs_times = np.empty(len(x_axis))
                    sp_times = np.empty(len(x_axis))
                    sort_times = np.empty(len(x_axis))
                    for i, nnz in enumerate(x_axis):
                        NNZ = nnz

                        out = index_time(SORT, N_THREADS,
                                                      SPARSE_FORMAT, ROWS,
                                                      COLS, NNZ, N_INDEXERS,
                                                      SEARCH_TYPE,
                                                      OPERATION)
                        cs_time, sp_time, sort_time = out

                        cs_times[i] = cs_time
                        sp_times[i] = sp_time
                        sort_times[i] = sort_time

                    # Save the info for plotting later.
                    plt_info.append({'x_axis': x_axis,
                                     'cs_times': cs_times,
                                     'sp_times': sp_times,
                                     'sort_times': sort_times,
                                     'SORT': SORT,
                                     'SPARSE_FORMAT': SPARSE_FORMAT,
                                     'OPERATION': OPERATION,
                                     'SEARCH_TYPE': SEARCH_TYPE,
                                     'N_THREADS': N_THREADS,
                                     'ROWS': ROWS,
                                     'COLS': COLS,
                                     'NNZ': NNZ,
                                     'N_INDEXERS': N_INDEXERS,
                                     'name': 'NNZ'})


    # Plot the times.
    plot_times(plt_info, ['NNZ'])


def test3():
    ## Test 3: How time changes on single thread, ROWS = COLS = 10000, with
    ##         fixed NNZ and varying N_INDEXERS.
    N_THREADS = 1
    ROWS = 10000
    COLS = 10000
    NNZ = 1000000
    x_axis = [100, 1000, 10000, 100000, 1000000, 10000000]

    plt_info = []
    for SORT in [True, False]:
        for SPARSE_FORMAT in ['CSR', 'CSC']:
            for OPERATION in ['get', 'add']:
                for SEARCH_TYPE in ['binary', 'interpolation', 'joint']:
                    # Create arrays for holding timing values for csindexer and
                    # for scipy indexer.
                    cs_times = np.empty(len(x_axis))
                    sp_times = np.empty(len(x_axis))
                    sort_times = np.empty(len(x_axis))
                    for i, n_indexers in enumerate(x_axis):
                        N_INDEXERS = n_indexers

                        out = index_time(SORT, N_THREADS,
                                                      SPARSE_FORMAT, ROWS,
                                                      COLS, NNZ, N_INDEXERS,
                                                      SEARCH_TYPE,
                                                      OPERATION)
                        cs_time, sp_time, sort_time = out

                        cs_times[i] = cs_time
                        sp_times[i] = sp_time
                        sort_times[i] = sort_time

                    # Save the info for plotting later.
                    plt_info.append({'x_axis': x_axis,
                                     'cs_times': cs_times,
                                     'sp_times': sp_times,
                                     'sort_times': sort_times,
                                     'SORT': SORT,
                                     'SPARSE_FORMAT': SPARSE_FORMAT,
                                     'OPERATION': OPERATION,
                                     'SEARCH_TYPE': SEARCH_TYPE,
                                     'N_THREADS': N_THREADS,
                                     'ROWS': ROWS,
                                     'COLS': COLS,
                                     'NNZ': NNZ,
                                     'N_INDEXERS': N_INDEXERS,
                                     'name': 'N_INDEXERS'})


    # Plot the times.
    plot_times(plt_info, ['N_INDEXERS'])

if __name__ == "__main__":
    #out = index_time(False, 1, 'CSR', 100000, 100000, 10000000, 10000000, 'binary', 'get')
    #print(out)
    test3()
