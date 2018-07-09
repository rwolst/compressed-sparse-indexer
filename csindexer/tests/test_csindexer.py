import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
from contexttimer import Timer
import pytest

from csindexer import indexer as csindexer

SORT = False
DEBUG = False  # Saves the large objects to csv for testing in C.
N_THREADS = 8

@pytest.fixture(scope="module")
def small_matrix():
    # Build small matrix
    matrix = [[ 0.  ,  0.  ,  0.45],
              [ 0.22,  0.74,  0.87],
              [ 0   ,  0   ,  0   ],
              [ 0   ,  0.6 ,  0   ],
              [ 0.  ,  0.93,  0.  ]]

    M= {}
    M['CSR'] = sp.sparse.csr_matrix(matrix)
    M['CSC'] = sp.sparse.csc_matrix(matrix)

    indexer = {}
    indexer['row'] = np.array([0, 0, 1, 1, 4, 4, 4], dtype=np.int32)
    indexer['col'] = np.array([2, 2, 0, 1, 1, 1, 1], dtype=np.int32)
    indexer['data'] = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    out = {'M': M, 'indexer': indexer}

    return out


@pytest.fixture(scope="module")
def large_matrix():
    # Build large matrix
    matrix = sp.sparse.rand(40000, 40000, density=0.001)

    M = {}
    M['CSR'] = sp.sparse.csr_matrix(matrix)
    M['CSC'] = sp.sparse.csc_matrix(matrix)

    # Indices must be a choice of the indices in matrix
    indexer = {}
    idx = np.random.choice(matrix.nnz, 100000, replace=True)
    indexer['row'] = matrix.row[idx]
    indexer['col'] = matrix.col[idx]
    indexer['data'] = np.random.rand(idx.size).astype(np.float64)

    if DEBUG:
        # Save the CSR matrix.
        data_dir = "csindexer/tests/data/"
        np.savetxt(data_dir + "data.csv", M['CSR'].data, delimiter=",")
        np.savetxt(data_dir + "indices.csv", M['CSR'].indices, delimiter=",")
        np.savetxt(data_dir + "indptr.csv", M['CSR'].indptr, delimiter=",")

        # Save the indexing objects.
        np.savetxt(data_dir + "row_vec.csv", indexer['row'], delimiter=",")
        np.savetxt(data_dir + "col_vec.csv", indexer['col'], delimiter=",")
        np.savetxt(data_dir + "data_vec.csv", indexer['data'], delimiter=",")

    out = {'M': M, 'indexer': indexer}

    return out


@pytest.mark.parametrize("SEARCH_TYPE", ['binary', 'interpolation', 'joint'])
def test_get_small(SEARCH_TYPE, small_matrix):
    print('\nGet small (%s):' % SEARCH_TYPE)
    M = small_matrix['M']
    indexer = small_matrix['indexer']
    true = np.array([0.45, 0.45, 0.22, 0.74, 0.93, 0.93, 0.93])

    for key in M:
        print('\n%s matrix' % key)

        with Timer() as t:
            if key == 'CSR':
                # Sort indices according to row first
                sort_idx = np.lexsort((indexer['col'], indexer['row']))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((indexer['row'], indexer['col']))

            # Technically don;t need to sort with binary search.
            if not SORT:
                sort_idx = np.arange(indexer['row'].size)
            unsort_idx = np.argsort(sort_idx)
        print('\tLexsort time: %s' % t.elapsed)

        with Timer() as t:
            data_cy = np.empty(indexer['row'].size, dtype=np.float64)
            start = t.elapsed
            csindexer.apply(M[key],
                          indexer['row'][sort_idx],
                          indexer['col'][sort_idx],
                          data_cy,
                          'get',
                          SEARCH_TYPE,
                          N_THREADS,
                            True)
            print('\tCython function time: %s' % (t.elapsed - start))

            # Unsort data_cy
            data_cy = data_cy[unsort_idx]

        print('\tCython time to get: %s' % t.elapsed)

        with Timer() as t:
            data_py = np.squeeze(np.array(M[key][indexer['row'][sort_idx],
                                                       indexer['col'][sort_idx]]))
            data_py = data_py[unsort_idx]
        print('\tPython time to get: %s' % t.elapsed)

        assert(np.all((data_cy - true)**2 < 1e-6))
        assert(np.all((data_py - true)**2 < 1e-6))


@pytest.mark.parametrize("SEARCH_TYPE", ['binary', 'interpolation', 'joint'])
def test_get_large(SEARCH_TYPE, large_matrix):
    print('\nGet large (%s):' % SEARCH_TYPE)
    M = large_matrix['M']
    indexer = large_matrix['indexer']

    for key in M:
        print('\n%s matrix' % key)

        with Timer() as t:
            if key == 'CSR':
                # Sort indices according to row first
                sort_idx = np.lexsort((indexer['col'], indexer['row']))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((indexer['row'], indexer['col']))

            # Technically don;t need to sort with binary search.
            if not SORT:
                sort_idx = np.arange(indexer['row'].size)
            unsort_idx = np.argsort(sort_idx)
        print('\tLexsort time: %s' % t.elapsed)

        with Timer() as t:
            data_cy = np.empty(indexer['row'].size, dtype=np.float64)

            start = t.elapsed
            csindexer.apply(M[key], 
                          np.array(indexer['row'][sort_idx]), 
                          np.array(indexer['col'][sort_idx]),
                          data_cy,
                          'get',
                          SEARCH_TYPE,
                          N_THREADS,
                            True)
            print('\tCython function time: %s' % (t.elapsed - start))

            # Unsort data_cy
            data_cy = data_cy[unsort_idx]

        print('\tCython time to get: %s' % t.elapsed)

        with Timer() as t:
            data_py = np.squeeze(np.array(M[key][indexer['row'][sort_idx],
                                                 indexer['col'][sort_idx]]))
            data_py = data_py[unsort_idx]
        print('\tPython time to get: %s' % t.elapsed)


    assert(np.all((data_cy - data_py)**2 < 1e-6))


@pytest.mark.parametrize("SEARCH_TYPE", ['binary', 'interpolation', 'joint'])
def test_add_small(SEARCH_TYPE, small_matrix):
    print('\nAdd small (%s):' % SEARCH_TYPE)
    M = small_matrix['M']
    indexer = small_matrix['indexer']

    true = {}
    true['CSR'] = np.array([2.45, 1.22, 1.74, 0.87, 0.6, 3.93])
    true['CSC'] = np.array([1.22, 1.74, 0.6, 3.93, 2.45, 0.87])

    for key in M:
        print('\n%s matrix' % key)
        M_copy_cy = M[key].copy()
        M_copy_py = M[key].copy()

        with Timer() as t:
            if key == 'CSR':
                # Sort indices according to row first
                sort_idx = np.lexsort((indexer['col'], indexer['row']))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((indexer['row'], indexer['col']))
            # Technically don;t need to sort with binary search.
            if not SORT:
                sort_idx = np.arange(indexer['row'].size)
        print('\tLexsort time: %s' % t.elapsed)

        with Timer() as t:
            start = t.elapsed
            csindexer.apply(M_copy_cy, 
                          indexer['row'][sort_idx], 
                          indexer['col'][sort_idx],
                          indexer['data'][sort_idx],
                          'add',
                          SEARCH_TYPE,
                          N_THREADS,
                            True)
            print('\tCython function time: %s' % (t.elapsed - start))

        print('\tCython time to add: %s' % t.elapsed)

        with Timer() as t:
            #import pdb; pdb.set_trace()
            df = pd.DataFrame(np.concatenate([indexer['row'][:,None], 
                                              indexer['col'][:,None], 
                                              indexer['data'][:,None]], axis=1), 
                    columns=['Row', 'Col', 'Data'])
            df = df.groupby(['Row', 'Col']).sum()
            df_row = df.index.get_level_values('Row').astype(np.int32)
            df_col = df.index.get_level_values('Col').astype(np.int32)
            df_data = df['Data']
        print('\tPandas groupby time: %s' % t.elapsed)

        with Timer() as t:
            M_copy_py[df_row, df_col] = M_copy_py[df_row, df_col] + df_data.as_matrix()

        print('\tPython time to add: %s' % t.elapsed)

        assert(np.all((M_copy_cy.data - true[key])**2 < 1e-6))
        assert(np.all((M_copy_py.data - true[key])**2 < 1e-6))


@pytest.mark.parametrize("SEARCH_TYPE", ['binary', 'interpolation', 'joint'])
def test_add_large(SEARCH_TYPE, large_matrix):
    print('\nAdd large (%s):' % SEARCH_TYPE)
    M = large_matrix['M']
    indexer = large_matrix['indexer']

    for key in M:
        print('\n%s matrix' % key)
        M_copy_cy = M[key].copy()
        M_copy_py = M[key].copy()

        with Timer() as t:
            if key == 'CSR':
                # Sort indices according to row first
                sort_idx = np.lexsort((indexer['col'], indexer['row']))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((indexer['row'], indexer['col']))
            # Technically don;t need to sort with binary search.
            if not SORT:
                sort_idx = np.arange(indexer['row'].size)
        print('\tLexsort time: %s' % t.elapsed)

        with Timer() as t:
            start = t.elapsed
            csindexer.apply(M_copy_cy, 
                          indexer['row'][sort_idx], 
                          indexer['col'][sort_idx],
                          indexer['data'][sort_idx],
                          'add',
                          SEARCH_TYPE,
                          N_THREADS,
                            True)
            print('\tCython function time: %s' % (t.elapsed - start))

        print('\tCython time to add: %s' % t.elapsed)

        with Timer() as t:
            #import pdb; pdb.set_trace()
            df = pd.DataFrame(np.concatenate([indexer['row'][:,None],
                                              indexer['col'][:,None],
                                              indexer['data'][:,None]], axis=1),
                    columns=['Row', 'Col', 'Data'])
            df = df.groupby(['Row', 'Col']).sum()
            df_row = df.index.get_level_values('Row').astype(np.int32)
            df_col = df.index.get_level_values('Col').astype(np.int32)
            df_data = df['Data']
        print('\tPandas groupby time: %s' % t.elapsed)

        with Timer() as t:
            M_copy_py[df_row, df_col] = M_copy_py[df_row, df_col] + df_data.as_matrix()

        print('\tPython time to add: %s' % t.elapsed)

        assert(np.all((M_copy_cy.data - M_copy_py.data)**2 < 1e-6))
        assert(np.all((M_copy_cy.indptr - M_copy_py.indptr)**2 < 1e-6))
        assert(np.all((M_copy_cy.indices - M_copy_py.indices)**2 < 1e-6))
