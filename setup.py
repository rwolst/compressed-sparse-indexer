#setup.py

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = []

# Indexer
ext_modules += [
        Extension("csindexer.indexer",
            sources=["./csindexer/indexer.pyx",
                     "./csindexer/indexer_c.c"],
            include_dirs=[numpy.get_include()],
            language='c',
            libraries=[]
            )
        ]

# setup
setup(
  name="csindexer",
  packages=["csindexer"],
  cmdclass={'build_ext': build_ext},
  ext_modules=ext_modules
)

