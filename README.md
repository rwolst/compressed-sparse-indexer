# Compressed Sparse Indexer
Tools for fast indexing in CSC and CSR matrices as well as a Cython interface
into scipy.sparse.

## Installation
Recommended to install in a virtual environment i.e. from the repository root:

    python3 -m venv env
    source env/bin/activate

Install requirements

    python3 -m pip install -r requirements.txt

Install the program

    python3 -m pip install -U .

You can now run the tests to check the installation by installing `pytest`

    python3 -m pip install pytest

and running

    python3 -m pytest

## Usage
All experiments can be run through the `main.py` end point. To see help about
the CLI interface and the different varaibles to the program, run

    python3 main.py --help

### Examples
Below we specify we want `n` (the size of both rows and columns) on the x-axis
against time taken and that we want separate graphs for each `search_type`:

- `binary`
- `interpolation`
- `joint`
- `scipy`

The other variables are constant.

    python3 main.py n --n-threads 1 --nnz 1000 --n-indexers 10000 \
        --n 50 100  500 1000 5000 10000 50000 100000 1000000 \
        --sort 0 --sparse-format CSR --operation get \
        --search-type binary interpolation joint scipy

<img src="figures/fig1.png" width="1200" height="1200">

In the next example we ensure that our indexer is sorted (`--sort 1`) which
allows us to test the optimised `--search-type sorted` algorithm as well.

    python3 main.py n --n-threads 1 --nnz 100000 --n-indexers 10000 \
        --n 1000 5000 10000 50000 100000 1000000 --sort 1 \
        --sparse-format CSR --operation get \
        --search-type binary interpolation joint scipy sorted

<img src="figures/fig2.png" width="1200" height="1200">

The final example shows that `scipy` seems to have a better complexity
'constant' with respect to `n-indexers` but grows at a faster rate with respect
to `n`.

    python3 main.py n --n-threads 1 --nnz 10000 --n-indexers 100000 \
        --n 1000 5000 10000 50000 100000 1000000 --sort 1 \
        --sparse-format CSR --operation get \
        --search-type binary interpolation joint scipy sorted

<img src="figures/fig3.png" width="1200" height="1200">

Finally for an unsorted version with larger `n-indexers` the `scipy` method
appears to dominate.

    python3 main.py n --n-threads 1 --nnz 10000 --n-indexers 100000 \
        --n 1000 5000 10000 50000 100000 1000000 --sort 0 \
        --sparse-format CSR --operation get \
        --search-type binary interpolation joint scipy

<img src="figures/fig4.png" width="1200" height="1200">

Although this is probably because it is able to use more threads. The
(`n_threads`) currently only applies to the algorithms in this repository, not
to the Scipy indexer (which I beleive uses matrix multiplication and hence the
maximum number of threads in all cases). In fact, this seems to make sense when
we increase the allowed threads

    python3 main.py n --n-threads -1 --nnz 10000 --n-indexers 100000 \
        --n 1000 5000 10000 50000 100000 1000000 --sort 0 \
        --sparse-format CSR --operation get \
        --search-type binary interpolation joint scipy

<img src="figures/fig5.png" width="1200" height="1200">

An interesting experiment that shows how increasing threads can go from worse
than Scipy to better than it

    python3 main.py n_threads --n-threads 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --nnz 1000000 --n-indexers 10000000 --n 10000 --sort 0 --sparse-format CSR --operation get --search-type binary interpolation joint scipy

<img src="figures/fig6.png" width="1200" height="1200">
