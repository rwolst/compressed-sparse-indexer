#setup.py

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = []

# Indexer
ext_modules += [
        Extension("csindexer.indexer",
            sources=["./csindexer/indexer.pyx",
                     "./csindexer/indexer_c.c",
                     "./csindexer/interpolation_search.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-Ofast", "-lm", "-fopenmp"],
            extra_link_args=["-fopenmp"],
            language='c',
            libraries=[]
            )
        ]

# setup
setup(
  name="csindexer",
  packages=["csindexer"],
  cmdclass={'build_ext': build_ext},
  ext_modules=ext_modules,
  setup_requires=["pytest-runner"],
  tests_require=["pytest"]
)

