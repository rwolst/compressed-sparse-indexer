"""An endpoint to run the speed benchmarks from."""

import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib.pyplot as plt
import os
import argparse
import itertools
from contexttimer import Timer

from csindexer import indexer as csindexer


# Use absolute paths to avoid any issues.
project_dir = os.path.dirname(os.path.realpath(__file__))

# Create argument parser.
parser = argparse.ArgumentParser(description="Endpoint for running tests on"
                                             " the compressed sparse indexer.")
parser.add_argument('dependent',
                    type=str,
                    default='rows',
                    help="The varaible to use on the x-axis when plotting"
                         " against time.")
parser.add_argument('--sort',
                    type=int,
                    nargs='+',
                    default=[0],
                    help="Assume the indexer is sorted (1) or not (0).")
parser.add_argument('--n-threads',
                    type=int,
                    nargs='+',
                    default=[1],
                    help="Total threads to use. Set as -1 to use maximum.")
parser.add_argument('--sparse-format',
                    type=str,
                    nargs='+',
                    default=['CSR'],
                    help="Whether to use CSR or CSC storage format.")
parser.add_argument('--n',
                    type=int,
                    nargs='+',
                    default=[],
                    help="Total rows and columns of the sparse matrix, forcing"
                         " it to be square. This can be useful if we want to"
                         " change both rows and columns on the x-axis. Leave"
                         " as [] to ignore.")
parser.add_argument('--rows',
                    type=int,
                    nargs='+',
                    default=[100],
                    help="Total rows of the sparse matrix.")
parser.add_argument('--cols',
                    type=int,
                    nargs='+',
                    default=[100],
                    help="Total columns of the sparse matrix.")
parser.add_argument('--nnz',
                    type=int,
                    nargs='+',
                    default=[100],
                    help="Total non-zero values in sparse matrix.")
parser.add_argument('--n-indexers',
                    type=int,
                    nargs='+',
                    default=[100],
                    help="Total number of points in the indexer.")
parser.add_argument('--search-type',
                    type=str,
                    nargs='+',
                    default=['binary'],
                    help="Whether to use binary, interpolation or joint"
                         " search, or the scipy indexer.")
parser.add_argument('--operation',
                    type=str,
                    nargs='+',
                    default=['get'],
                    help="Whether to use a get or add operation.")
parser.add_argument('--save',
                    action='store_true',
                    help="Whether to save the plot to ./figures.")
parser.add_argument('--figure-name',
                    type=str,
                    default='my_figure.png',
                    help="What to call the plot.")
parser.add_argument('--debug',
                    action='store_true',
                    help="Print the configuration when creating model.")
parser.add_argument('--random-seed',
                    type=int,
                    default=np.random.randint(0, 2**32 - 1),
                    help="Value of random seed")

FLAGS, unparsed = parser.parse_known_args()
np.random.seed(FLAGS.random_seed)
config = FLAGS.__dict__.copy()


def index_time(sort, n_threads, sparse_format, rows, cols, nnz, n_indexers,
               search_type, operation, debug):
    """A function for timing our cxindexer and scipy indexer. It first creates
    sparse matrices, sorts if necessary, runs indexers on both and returns
    the times."""
    if debug:
        print("Benchmarking:\n\tSORT = %s\n\tN_THREADS = %s\n\tSPARSE_FORMAT ="
              " %s\n\tROWS = %s\n\tCOLS = %s\n\tNNZ = %s\n\tN_INDEXERS ="
              " %s\n\t" "SEARCH_TYPE = %s\n\tOPERATION = %s"
              % (sort, n_threads, sparse_format, rows, cols, nnz, n_indexers,
                 search_type, operation))

    # Generate matrix.
    with Timer() as t:
        M = sp.sparse.rand(rows, cols, density=nnz/(rows*cols))

    if debug:
        print("\tTime to generate sparse matrix: %s" % t.elapsed)

    # Generate indexer.
    with Timer() as t:
        indexer = {}
        idx = np.random.choice(M.nnz, n_indexers, replace=True)
        indexer['row'] = M.row[idx]
        indexer['col'] = M.col[idx]
        indexer['data'] = np.random.rand(idx.size).astype(np.float64)

    if debug:
        print("\tTime to generate indexer: %s" % t.elapsed)

    # Convert sparse matrix.
    with Timer() as t:
        if sparse_format == 'CSR':
            M = sp.sparse.csr_matrix(M)
        elif sparse_format == 'CSC':
            M = sp.sparse.csc_matrix(M)
        else:
            raise Exception("sparse_format must be either CSR or CSC.")

    if debug:
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

    if debug:
        print("\tTime to sort indexer: %s" % t.elapsed)
    sort_time = t.elapsed

    # Time the csindexer.
    with Timer() as t:
        if search_type == 'scipy':
            ## Run the Scipy function.
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
                else:
                    raise Exception("Operation must be either get or add.")

        else:
            ## Run the Cython function.
            if operation == 'get':
                ### Don't need to copy M as it doesn't get modified but do have
                ### to copy indexer['data'] as it does.
                data_cs = indexer['data'].copy()
                M_cs = M

                csindexer.apply(M_cs, np.array(indexer['row'][sort_idx]),
                                np.array(indexer['col'][sort_idx]), data_cs,
                                operation, search_type, n_threads, debug)

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
                                n_threads, debug)
            else:
                raise Exception("Operation must be either get or add.")


    if debug:
        print("\tTime for indexing: %s" % t.elapsed)
    computation_time = t.elapsed

    return computation_time, sort_time


if __name__ == "__main__":

    # Dependent variable gets plotted on x-axis, all others are separate lines
    # on the plot.

    # Get the list of separate models to be plotted.
    if config['n'] != []:
        # Override the rows and cols using n (so the sparse matrix is square).
        variables = ['sort', 'n_threads', 'sparse_format', 'n',
                     'nnz', 'n_indexers', 'search_type', 'operation']
    else:
        variables = ['sort', 'n_threads', 'sparse_format', 'rows', 'cols',
                     'nnz', 'n_indexers', 'search_type', 'operation']

    dependent = config['dependent']
    variables.remove(dependent)

    models = list(itertools.product(*[config[i] for i in variables]))

    # Convert models into dictionaries.
    models = [dict(zip(variables, i)) for i in models]

    # Now loop over the dependent variable and get timings for each model.
    times = np.empty([len(config[dependent]), len(models)])
    for i, x in enumerate(config[dependent]):
        for j, model in enumerate(models):
            m = model.copy()
            m[dependent] = x

            if 'n' in m:
                m['rows'] = m['n']
                m['cols'] = m['n']

            times[i, j], _ = index_time(m['sort'], m['n_threads'],
                                        m['sparse_format'], m['rows'],
                                        m['cols'], m['nnz'], m['n_indexers'],
                                        m['search_type'], m['operation'],
                                        config['debug'])

    # Finally plot each model.
    ## Get the maximum time seen.
    max_time = times.max()
    plt.figure(figsize=(20,20))
    plt.ylim(0, max_time)

    ## Plot each model.
    for j, model in enumerate(models):
        plt.plot(config[dependent], times[:, j])

    plt.xlabel(dependent)

    ## For the legend only use variables that have changed i.e. more than 1
    ## input.
    used_variables = [i for i in variables if len(config[i]) != 1]
    unused_variables = [i for i in variables if len(config[i]) == 1]

    models_legend = [dict(zip(used_variables,
                              [model[i] for i in used_variables]))
                     for model in models]
    plt.legend(models_legend)

    models_title = dict(zip(unused_variables,
                            [config[i] for i in unused_variables]))
    title = "Sparse indexing %s vs time (%s)" % (dependent, models_title)
    plt.title(title)
    if config['save']:
        fname = project_dir + '/figures/' + config['figure_name']
        plt.savefig(fname)

    plt.show()

