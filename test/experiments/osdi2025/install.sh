#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# set -x # Uncomment for debugging (print each command before execution)

echo "Starting QUAKE Artifact and Baselines Setup Script."
echo "This script should be run with sudo (e.g., 'sudo ./install.sh')."
echo "Targeting Ubuntu 22.04 (Jammy) based system."

# -----------------------------
# Environment Variables
# -----------------------------
export CONDA_DIR_PATH="/opt/miniconda"
# Ensure Miniconda's bin is in PATH for the root user running the script
export PATH="${CONDA_DIR_PATH}/bin:${PATH}"
export DEBIAN_FRONTEND=noninteractive

# -----------------------------
# Update System and Install Base APT Packages
# -----------------------------
echo ">>> Updating system and installing base APT packages..."
apt-get update

apt-get install -y --no-install-recommends \
    wget \
    curl \
    build-essential \
    ca-certificates \
    swig \
    git \
    libomp-dev \
    graphviz \
    numactl \
    libnuma-dev \
    htop \
    software-properties-common

# -----------------------------
# Install/Ensure GCC 11 and G++ 11
# -----------------------------
echo ">>> Ensuring GCC 11 and G++ 11 are set up correctly..."
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y gcc-11 g++-11

echo ">>> Setting up GCC and G++ alternatives to point to version 11..."
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
                         --slave /usr/bin/gcov gcc-gcov /usr/bin/gcov-11 \
                         --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 \
                         --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-11 \
                         --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
update-alternatives --set gcc /usr/bin/gcc-11
update-alternatives --set g++ /usr/bin/g++-11

echo "Verifying GCC version after update-alternatives:"
gcc --version
echo "Verifying G++ version after update-alternatives:"
g++ --version

# -----------------------------
# Install CMake 3.24.2
# -----------------------------
CMAKE_VERSION_EXPECTED="3.24.2"
CMAKE_INSTALL_PATH="/usr/local/bin/cmake"
echo ">>> Installing CMake ${CMAKE_VERSION_EXPECTED}..."
if command -v cmake &> /dev/null && [[ "$(cmake --version)" == *"cmake version ${CMAKE_VERSION_EXPECTED}"* ]]; then
    echo "CMake version ${CMAKE_VERSION_EXPECTED} already installed at $(command -v cmake). Skipping."
else
    echo "Installing CMake ${CMAKE_VERSION_EXPECTED}..."
    cd /tmp
    wget -qO cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION_EXPECTED}/cmake-${CMAKE_VERSION_EXPECTED}-linux-x86_64.sh
    chmod +x cmake.sh
    ./cmake.sh --skip-license --prefix=/usr/local # Runs as root
    rm cmake.sh
    cd -
fi
echo "Verifying CMake version:"
cmake --version

# -----------------------------
# Install Miniconda
# -----------------------------
echo ">>> Installing Miniconda to ${CONDA_DIR_PATH}..."
if [ -x "${CONDA_DIR_PATH}/bin/conda" ]; then
    echo "Miniconda already found in ${CONDA_DIR_PATH}. Skipping installation."
else
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "${CONDA_DIR_PATH}" # Runs as root, installs to /opt/miniconda
    rm miniconda.sh
    cd -
    echo ">>> Initializing Conda for bash (modifies /root/.bashrc as script runs with sudo)..."
    "${CONDA_DIR_PATH}/bin/conda" init bash
fi

# Source conda's profile script to make conda command available in this script session
echo ">>> Sourcing Conda profile script for current root session..."
if [ -f "${CONDA_DIR_PATH}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR_PATH}/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda profile script not found at ${CONDA_DIR_PATH}/etc/profile.d/conda.sh after installation attempt."
    echo "Conda commands might not be available. Exiting."
    exit 1
fi

# -----------------------------
# Create Conda Environment 'quake-env' and Install Dependencies
# -----------------------------
CONDA_ENV_NAME="quake-env"
echo ">>> Setting up conda environment '${CONDA_ENV_NAME}'..."

echo ">>> Creating conda.yaml file at /tmp/conda.yaml..."
# Ensure the file is removed first to prevent issues with existing file/permissions
rm -f /tmp/conda.yaml
cat > /tmp/conda.yaml << EOF
name: ${CONDA_ENV_NAME}
channels:
  - pytorch
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
  - faiss-cpu
  - matplotlib
  - pytest
  - pip
  - pip:
    - sphinx
    - sphinx_rtd_theme
    - sphinxcontrib-mermaid
    - graphviz
    - pyyaml
EOF

# If environment exists, remove it for a clean setup
if conda env list | grep -E "^${CONDA_ENV_NAME}\s+"; then
    echo "Conda environment '${CONDA_ENV_NAME}' already exists. Removing for a clean setup..."
    conda env remove -n "${CONDA_ENV_NAME}" --all -y
fi
echo ">>> Creating conda environment '${CONDA_ENV_NAME}' from /tmp/conda.yaml..."
conda env create -f /tmp/conda.yaml
rm /tmp/conda.yaml # Clean up
conda clean -afy

echo ">>> Installing specific PyTorch (CPU) and other Python packages into '${CONDA_ENV_NAME}'..."
conda run -n "${CONDA_ENV_NAME}" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

conda run -n "${CONDA_ENV_NAME}" pip install \
    lightgbm \
    scikit-learn \
    scann==1.4.0

# Force numpy 1.25.0 for diskannpy compatibility
conda run -n "${CONDA_ENV_NAME}" pip install numpy==1.25.0 --force-reinstall
conda run -n "${CONDA_ENV_NAME}" pip install diskannpy==0.7.0

echo ">>> Verifying '${CONDA_ENV_NAME}'..."
conda env list
conda run -n "${CONDA_ENV_NAME}" python -c "import sys; print(f'OK in {sys.prefix}; python:', sys.executable); import torch; print('PyTorch version:', torch.__version__); import numpy; print('Numpy version:', numpy.__version__)"

# -----------------------------
# Install Intel oneAPI Base Toolkit
# -----------------------------
ONEAPI_INSTALL_PATH="/opt/intel/oneapi"
echo ">>> Installing Intel oneAPI Base Toolkit to ${ONEAPI_INSTALL_PATH}..."
if [ -f "${ONEAPI_INSTALL_PATH}/setvars.sh" ]; then
    echo "Intel oneAPI already found in ${ONEAPI_INSTALL_PATH}. Skipping installation."
else
    cd /tmp
    wget -O intel-oneapi-base-toolkit-offline.sh https://registrationcenter-download.intel.com/akdlm/IRC_NAS/6bfca885-4156-491e-849b-1cd7da9cc760/intel-oneapi-base-toolkit-2025.1.1.36_offline.sh
    chmod +x intel-oneapi-base-toolkit-offline.sh
    ./intel-oneapi-base-toolkit-offline.sh -a --silent --cli --eula accept --install-dir "${ONEAPI_INSTALL_PATH}" # sudo implied
    rm intel-oneapi-base-toolkit-offline.sh
    cd -
fi
# Source oneAPI environment variables for the current session.
echo ">>> Sourcing Intel oneAPI setvars.sh for current root session (needed for SVS build)..."
if [ -f "${ONEAPI_INSTALL_PATH}/setvars.sh" ]; then
    source "${ONEAPI_INSTALL_PATH}/setvars.sh"
else
    echo "WARNING: oneAPI setvars.sh not found at ${ONEAPI_INSTALL_PATH}/setvars.sh. SVS build might fail."
fi

# -----------------------------
# Clone and Install QUAKE Repository
# -----------------------------
QUAKE_DIR_BASE="/opt"
QUAKE_REPO_NAME="quake"
QUAKE_FULL_PATH="${QUAKE_DIR_BASE}/${QUAKE_REPO_NAME}"

echo ">>> Cloning/Updating and installing QUAKE repository in ${QUAKE_FULL_PATH}..."
if [ -d "${QUAKE_FULL_PATH}" ]; then
  echo "QUAKE directory ${QUAKE_FULL_PATH} already exists. Updating..."
  cd "${QUAKE_FULL_PATH}"
  git pull
else
  git clone https://github.com/marius-team/quake.git "${QUAKE_FULL_PATH}"
  cd "${QUAKE_FULL_PATH}"
fi
git checkout osdi2025_debug

git config --global --add safe.directory "${QUAKE_FULL_PATH}" # For root's global config
SUBMODULE_PATHS=("src/cpp/third_party/concurrentqueue" "src/cpp/third_party/faiss" "src/cpp/third_party/pybind11")
for SUBMODULE_PATH in "${SUBMODULE_PATHS[@]}"; do
    git config --global --add safe.directory "${QUAKE_FULL_PATH}/${SUBMODULE_PATH}"
done
git submodule update --init --recursive

echo ">>> Building and installing QUAKE Python package..."
conda run -n "${CONDA_ENV_NAME}" pip install . --no-use-pep517
cd /

# -----------------------------
# Clone and Install ScalableVectorSearch (SVS)
# -----------------------------
SVS_DIR_BASE="/opt"
SVS_REPO_NAME="ScalableVectorSearch"
SVS_FULL_PATH="${SVS_DIR_BASE}/${SVS_REPO_NAME}"

echo ">>> Cloning/Updating ScalableVectorSearch repository in ${SVS_FULL_PATH}..."
if [ -d "${SVS_FULL_PATH}" ]; then
  echo "ScalableVectorSearch directory ${SVS_FULL_PATH} already exists. Updating..."
  cd "${SVS_FULL_PATH}"
  git pull
else
  git clone https://github.com/intel/ScalableVectorSearch.git "${SVS_FULL_PATH}"
  cd "${SVS_FULL_PATH}"
fi
git config --global --add safe.directory "${SVS_FULL_PATH}"

echo ">>> Attempting to build and install SVS Python bindings..."
echo "This step requires GCC/G++ 11 and Intel oneAPI tools (which should be sourced)."
echo "Current compilers: GCC=$(which gcc || echo not_found), G++=$(which g++ || echo not_found)"
# The oneAPI setvars.sh should have configured the environment for Intel compilers if available and preferred by SVS build.
# Otherwise, it should use the g++-11 from update-alternatives.
conda run -n "${CONDA_ENV_NAME}" pip install ./bindings/python
cd /

echo "--------------------------------------------------------------------"
echo "Setup script finished successfully."
echo "To activate the conda environment (in a new shell, as root or user depending on .bashrc): conda activate ${CONDA_ENV_NAME}"
echo "QUAKE repository is in ${QUAKE_FULL_PATH}"
echo "Intel oneAPI (if installed) is in ${ONEAPI_INSTALL_PATH}. Source with: source ${ONEAPI_INSTALL_PATH}/setvars.sh"
echo "SVS repository (if cloned) is in ${SVS_FULL_PATH}"
echo "--------------------------------------------------------------------"
