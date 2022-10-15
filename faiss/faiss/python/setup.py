# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
import os
import shutil
import platform

# make the faiss python package dir
shutil.rmtree("faissAdaptive", ignore_errors=True)
os.mkdir("faissAdaptive")
# shutil.copytree("contrib", "faiss/contrib")
shutil.copyfile("__init__.py", "faissAdaptive/__init__.py")
shutil.copyfile("loader.py", "faissAdaptive/loader.py")

ext = ".pyd" if platform.system() == 'Windows' else ".so"
prefix = "Release/" * (platform.system() == 'Windows')

swigfaiss_generic_lib = f"{prefix}_swigfaiss{ext}"
swigfaiss_avx2_lib = f"{prefix}_swigfaiss_avx2{ext}"

found_swigfaiss_generic = os.path.exists(swigfaiss_generic_lib)
found_swigfaiss_avx2 = os.path.exists(swigfaiss_avx2_lib)

assert (found_swigfaiss_generic or found_swigfaiss_avx2), \
    f"Could not find {swigfaiss_generic_lib} or " \
    f"{swigfaiss_avx2_lib}. Faiss may not be compiled yet."

if found_swigfaiss_generic:
    print(f"Copying {swigfaiss_generic_lib}")
    shutil.copyfile("swigfaiss.py", "faissAdaptive/swigfaiss.py")
    shutil.copyfile(swigfaiss_generic_lib, f"faissAdaptive/_swigfaiss{ext}")

if found_swigfaiss_avx2:
    print(f"Copying {swigfaiss_avx2_lib}")
    shutil.copyfile("swigfaiss_avx2.py", "faissAdaptive/swigfaiss_avx2.py")
    shutil.copyfile(swigfaiss_avx2_lib, f"faissAdaptive/_swigfaiss_avx2{ext}")

long_description="""
Faiss is a library for efficient similarity search and clustering of dense
vectors. It contains algorithms that search in sets of vectors of any size,
 up to ones that possibly do not fit in RAM. It also contains supporting
code for evaluation and parameter tuning. Faiss is written in C++ with
complete wrappers for Python/numpy. Some of the most useful algorithms
are implemented on the GPU. It is developed by Facebook AI Research.
"""
setup(
    name='faissAdaptive',
    version='1.7.2',
    description='A library for efficient similarity search and clustering of dense vectors',
    long_description=long_description,
    url='https://github.com/facebookresearch/faiss',
    author='Matthijs Douze, Jeff Johnson, Herve Jegou, Lucas Hosseini',
    author_email='matthijs@fb.com',
    license='MIT',
    keywords='search nearest neighbors',

    install_requires=['numpy'],
    packages=['faissAdaptive'],
    package_data={
        'faissAdaptive': ['*.so', '*.pyd'],
    },
    zip_safe=False,
)
