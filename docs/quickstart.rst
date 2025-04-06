Quick Start Guide
=================

Welcome to Quake! This guide provides you with the essential steps to quickly set up, build, and run Quake—a high‑performance dynamic vector search engine with Python bindings.

Prerequisites
-------------
Before you begin, ensure you have the following installed:

- **C++17 Compiler** (e.g. GCC 8+, Clang 10+, or MSVC 2019+)
- **Python 3.9+**
- **Conda** (for dependency management)
- **Git**

Step 1: Clone the Repository
-----------------------------
Clone the repository and initialize submodules:

.. code-block:: bash

    git clone https://github.com/marius-team/quake.git
    cd quake
    git submodule update --init --recursive

Step 2: Set Up the Environment
------------------------------
Create and activate the Conda environment using the provided YAML file:

.. code-block:: bash

    conda env create -f environments/ubuntu-latest/conda.yaml
    conda activate quake-env

Step 3: Install Quake
---------------------
Install the Quake package (which includes the Python bindings):

.. code-block:: bash

    pip install --no-use-pep517 .

Step 4: Run Example Program
-------------------------------
Quake comes with an example program that demonstrates its basic functionality.
Run the example by executing the following command:

.. code-block:: bash

    python examples/quickstart.py

This program will:
- Build an index from a sample dataset.
- Perform a basic search.
- Print the search results along with performance metrics.

For more detailed instructions, please refer to the **Installation Guide** (:doc:`install`) and the **Developer Guide** (:doc:`development_guide`).