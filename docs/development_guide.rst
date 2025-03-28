Development Guide
=======================

Quake is a high‑performance vector search engine written in C++ with Python bindings. It supports adaptive, real‑time updates and query‑adaptive search—letting you specify a recall target so that the index automatically adjusts its search scope. This guide will help you rapidly understand our design, coding standards and contribution workflow.

Codebase Overview
----------------------------
Quake’s design is split into two major layers:

1. **The C++ Core**
   Implements the heavy‑lifting: index construction, query processing, vector updates, and dynamic maintenance (splits, merges, and reassignments). It is organized in the ``src/cpp/`` directory. Key components include:

   - **QuakeIndex:**
     The central class that coordinates building, searching, and maintaining the index.
   - **PartitionManager:**
     Manages dynamic partitions (or “clusters”) of vectors.
   - **MaintenancePolicy:**
     Encapsulates the rules for when and how to split or merge partitions to meet recall targets.
   - **QueryCoordinator:**
     Distributes search queries across partitions and aggregates results.
   - **Bindings:**
     C++ functionality is exposed to Python via pybind11 (located in ``src/cpp/bindings``).

2. **The Python Layer**
   Provides user-friendly wrappers, dataset loaders, and utility functions for integrating with PyTorch and other ML workflows. It is located in the ``src/python/`` directory and uses Sphinx (with autodoc) to extract docstrings from our Python code.


See the :doc:`architecture/architecture` for a detailed breakdown of the components and their interactions.


Directory Structure
-------------------
Familiarize yourself with the layout of the repository:

.. code-block:: text

    .
    ├── CMakeLists.txt              # CMake build configuration
    ├── README.md                   # High-level project description
    ├── docs/                       # Documentation sources (RST files, Sphinx config)
    │   ├── index.rst
    │   ├── install.rst
    │   └── development_guide.rst    <-- This guide
    ├── src/
    │   ├── cpp/                    # C++ source, headers, and third‑party submodules
    │   │   ├── include/            # Public headers (API)
    │   │   ├── src/                # Implementation files
    │   │   ├── bindings/           # Python bindings via pybind11
    │   │   └── third_party/        # External dependencies (e.g., Faiss, SimSIMD)
    │   └── python/                 # Python modules and utilities
    ├── test/                      # Unit and integration tests (C++ and Python)
    ├── setup.py / setup.cfg       # Python packaging files
    └── .gitmodules                # Git submodule configuration

Contribution Workflow & Coding Standards
------------------------------------------
We expect all contributors to follow these practices:

Git Workflow & PRs
******************

- **Branching:** Create feature branches from the main branch.
- **Pull Requests:** Submit clear PRs with detailed descriptions and links to related issues.
- **Code Reviews:** Expect direct feedback—clarity and correctness are our top priorities.

Coding Standards
****************

- **C++:** Follow the `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_
- **Python:** Adhere to `PEP8 <https://peps.python.org/pep-0008/>`_
- **Docstrings & Comments:** Every class and function should be documented. Clear inline comments and comprehensive docstrings help both human readers and our automated documentation tools.

Testing
**********

- **C++ Tests:** Located in ``test/cpp/``; run them via CMake (e.g. using ``ctest`` or ``make quake_tests``).

- **Python Tests:** Located in ``test/python/``; run them with pytest.

- **When Adding Features:** Always add tests covering new functionality and ensure tests are clear and reflect real usage scenarios.

Workflow
--------------------------

1. **Fork and Set Up:**

.. code-block:: bash

   git fork https://github.com/marius-team/quake.git
   cd quake
   git submodule update --init --recursive

2. **Create a Feature Branch:**

.. code-block:: bash

   git checkout -b feature/my-feature

3. **Build and Activate Conda Environment:**

This installs the necessary dependencies for building Quake. See :doc:`install` for more details.

.. code-block:: bash

   conda env create -f environments/ubuntu-latest/conda.yaml
   conda activate quake-env

4. **Build the Code & Bindings:**

C++ Build (optional, if you only want to work on Python code):

.. code-block:: bash

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DQUAKE_ENABLE_GPU=ON \
          -DQUAKE_USE_NUMA=ON \
          -DQUAKE_USE_AVX512=ON \
          -DBUILD_TESTS=ON ..
    make -j$(nproc) bindings

Python Build

.. code-block:: bash

   pip install --no-use-pep517 .


5. **Run Tests:**

**C++ Tests:**

Build the tests and run them (assuming you are in the ``build/`` directory):

.. code-block:: bash

 make -j$(nproc) quake_tests
 test/cpp/quake_tests --gtest_filter=* # use filters to run specific tests

**Python Tests:**

Quake must be installed with pip to run the Python tests. Run them using pytest:

.. code-block:: bash

 pytest test/python/

6. **Run Autoformatting and Linters:**

Lint checks need to pass before submitting a PR.

We use `black`, `isort` and `flake8`.

Run the following scripts to autoformat and run linters:

.. code-block:: bash

   source scripts/autoformat.sh
   source scripts/lint.sh

7. **Make Changes and submit a PR:**

After making changes, commit them and push to your branch. Then, create a PR on the main branch.
