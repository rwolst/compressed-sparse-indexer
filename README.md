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
Below we specify we want `n` (the size of both rows and columns) on the x-axis
against time taken and that we want separate graphs for each `search_type`:

- `binary`
- `interpolation`
- `joint`
- `scipy`

The other variables are constant.

    python3 main.py n --n-threads 1 --nnz 1000 --n-indexers 10000 --n 50 100  500 1000 5000 10000 50000 100000 1000000 --sort 0 --sparse-format CSR --operation get --search-type binary interpolation joint scipy

In the next example we ensure that our indexer is sorted (`--sort 1`) which
allows us to test the optimised `--search-type sorted` algorithm as well.

    python3 main.py n --n-threads 1 --nnz 100000 --n-indexers 10000 --n 1000 5000 10000 50000 100000 1000000 --sort 1 --sparse-format CSR --operation get --search-type binary interpolation joint scipy sorted

The final example shows that `scipy` seems to have a better complexity
'constant' with respect to `n-indexers` but grows at a faster rate with respect
to `n`.

    python3 main.py n --n-threads 1 --nnz 10000 --n-indexers 100000 --n 1000 5000 10000 50000 100000 1000000 --sort 1 --sparse-format CSR --operation get --search-type binary interpolation joint scipy sorted

Finally for an unsorted version with larger `n-indexers` the `scipy` method
appears to dominate.

    python3 main.py n --n-threads 1 --nnz 10000 --n-indexers 100000 --n 1000 5000 10000 50000 100000 1000000 --sort 0 --sparse-format CSR --operation get --search-type binary interpolation joint scipy

although this is probably because it is able to use more threads (`n_threads`)
currently only applies to the algorithms in this repository. In fact, this
seems to make sense when we increase the allowed threads

    python3 main.py n --n-threads -1 --nnz 10000 --n-indexers 100000 --n 1000 5000 10000 50000 100000 1000000 --sort 0 --sparse-format CSR --operation get --search-type binary interpolation joint scipy
