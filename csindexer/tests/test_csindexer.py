import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
from contexttimer import Timer

from csindexer import indexer

# Build small matrix
matrix = [[ 0.  ,  0.  ,  0.45],
          [ 0.22,  0.74,  0.87],
          [ 0   ,  0   ,  0   ],
          [ 0   ,  0.6 ,  0   ],
          [ 0.  ,  0.93,  0.  ]]

M_small = {}
M_small['CSR'] = sp.sparse.csr_matrix(matrix)
M_small['CSC'] = sp.sparse.csc_matrix(matrix)

row_vector_small = np.array([0, 0, 1, 1, 4, 4, 4], dtype=np.int32)
col_vector_small = np.array([2, 2, 0, 1, 1, 1, 1], dtype=np.int32)
indices_small = np.concatenate([row_vector_small[:, None], 
                                col_vector_small[:, None]], axis=1)
data_vector_small = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float64)


# Build large matrix
matrix = sp.sparse.rand(40000, 40000, density=0.001)

M_large = {}
M_large['CSR'] = sp.sparse.csr_matrix(matrix)
M_large['CSC'] = sp.sparse.csc_matrix(matrix)

# Indices must be a choice of the indices in matrix
idx = np.random.choice(matrix.nnz, 1000000, replace=True)
row_vector_large = matrix.row[idx]
col_vector_large = matrix.col[idx]
indices_large = np.concatenate([row_vector_large[:, None], 
                                col_vector_large[:, None]], axis=1)
data_vector_large = np.random.rand(idx.size).astype(np.float64)


def test_get_small():
    print('\nGet small:')
    true = np.array([0.45, 0.45, 0.22, 0.74, 0.93, 0.93, 0.93])
    
    for key in M_small:
        print('\n%s matrix' % key)

        with Timer() as t:
            if key == 'CSR':
                # Sort indices according to row first
                sort_idx = np.lexsort((col_vector_small, row_vector_small))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((row_vector_small, col_vector_small))
        print('\tLexsort time: %s' % t.elapsed)
        
        with Timer() as t:
            data_cy = np.empty(row_vector_small.size, dtype=np.float64)
            indexer.apply(M_small[key], 
                          row_vector_small[sort_idx], 
                          col_vector_small[sort_idx],
                          data_cy,
                          'get')

            # Unsort data_cy
            unsort_idx = np.argsort(sort_idx)
            data_cy = data_cy[unsort_idx]

        print('\tCython time to get: %s' % t.elapsed)

        with Timer() as t:
            data_py = np.squeeze(np.array(M_small[key][row_vector_small, 
                                                       col_vector_small]))
        print('\tPython time to get: %s' % t.elapsed)

        assert(np.all((data_cy - true)**2 < 1e-6))
        assert(np.all((data_py - true)**2 < 1e-6))


def test_get_large():
    print('\nGet large:')
    for key in M_large:
        print('\n%s matrix' % key)

        with Timer() as t:
            if key == 'CSR':
                # Sort indices according to row first
                sort_idx = np.lexsort((col_vector_large, row_vector_large))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((row_vector_large, col_vector_large))
        print('\tLexsort time: %s' % t.elapsed)
        
        with Timer() as t:
            data_cy = np.empty(row_vector_large.size, dtype=np.float64)
            indexer.apply(M_large[key], 
                          np.array(row_vector_large[sort_idx]), 
                          np.array(col_vector_large[sort_idx]),
                          data_cy,
                          'get')

            # Unsort data_cy
            unsort_idx = np.argsort(sort_idx)
            data_cy = data_cy[unsort_idx]

        print('\tCython time to get: %s' % t.elapsed)

        with Timer() as t:
            data_py = np.squeeze(np.array(M_large[key][row_vector_large, col_vector_large]))
        print('\tPython time to get: %s' % t.elapsed)


    assert(np.all((data_cy - data_py)**2 < 1e-6))

def test_add_small():
    print('\nAdd small:')
    true = {}
    true['CSR'] = np.array([2.45, 1.22, 1.74, 0.87, 0.6, 3.93])
    true['CSC'] = np.array([1.22, 1.74, 0.6, 3.93, 2.45, 0.87])

    for key in M_small:
        print('\n%s matrix' % key)
        M_copy_cy = M_small[key].copy()
        M_copy_py = M_small[key].copy()

        with Timer() as t:
            if key == 'CSR':
                # Sort indices according to row first
                sort_idx = np.lexsort((col_vector_small, row_vector_small))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((row_vector_small, col_vector_small))
        print('\tLexsort time: %s' % t.elapsed)
        
        with Timer() as t:
            indexer.apply(M_copy_cy, 
                          row_vector_small[sort_idx], 
                          col_vector_small[sort_idx],
                          data_vector_small[sort_idx],
                          'add')

        print('\tCython time to add: %s' % t.elapsed)

        with Timer() as t:
            #import pdb; pdb.set_trace()
            df = pd.DataFrame(np.concatenate([row_vector_small[:,None], 
                                              col_vector_small[:,None], 
                                              data_vector_small[:,None]], axis=1), 
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

def test_add_large():
    print('\nAdd large:')
    for key in M_small:
        print('\n%s matrix' % key)
        M_copy_cy = M_large[key].copy()
        M_copy_py = M_large[key].copy()

        with Timer() as t:
            if key == 'CSR':
                # Sort indices according to row first
                sort_idx = np.lexsort((col_vector_large, row_vector_large))
            else:
                # Sort indices according to col first
                sort_idx = np.lexsort((row_vector_large, col_vector_large))
        print('\tLexsort time: %s' % t.elapsed)
        
        with Timer() as t:
            indexer.apply(M_copy_cy, 
                          row_vector_large[sort_idx], 
                          col_vector_large[sort_idx],
                          data_vector_large[sort_idx],
                          'add')

        print('\tCython time to add: %s' % t.elapsed)

        with Timer() as t:
            #import pdb; pdb.set_trace()
            df = pd.DataFrame(np.concatenate([row_vector_large[:,None], 
                                              col_vector_large[:,None], 
                                              data_vector_large[:,None]], axis=1), 
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
