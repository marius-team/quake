Installation
============

This document explains how to install Quake on your system.

Prerequisites
-------------
- A C++17–compliant compiler (e.g. GCC 8+, Clang 10+)
- CMake 3.16 or newer
- Python 3.9+
- PyTorch 2.0+

Building from source
--------------------
Clone the repository and build the extension:

.. code-block:: bash

    git clone https://github.com/marius-team/quake.git
    cd quake
    git submodule update --init --recursive
    conda env create -f environments/ubuntu-latest/conda.yaml
    conda activate quake-env
    pip install --no-use-pep517 .

For advanced build options (e.g. enabling GPU, NUMA, or AVX512), use the cmake build:

.. code-block:: bash

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DQUAKE_ENABLE_GPU=ON \
          -DQUAKE_USE_NUMA=ON \
          -DQUAKE_USE_AVX512=ON ..
    make bindings -j$(nproc)
