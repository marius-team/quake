FROM ubuntu:20.04

# -----------------------------
# Set up environment variables
# -----------------------------
ENV CONDA_DIR=/opt/miniconda
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      curl \
      build-essential \
      ca-certificates \
      swig \
      git \
      libomp5 \
      libomp-dev \
      graphviz \
    && rm -rf /var/lib/apt/lists/*


RUN wget -qO /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1-linux-x86_64.sh && \
    chmod +x /tmp/cmake.sh && \
    /tmp/cmake.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake.sh

# -----------------------------
# Install Miniconda
# -----------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# -----------------------------
# Set up and activate conda environment
# -----------------------------
COPY environments/ubuntu-latest/conda.yaml /tmp/conda.yaml
RUN conda env create -f /tmp/conda.yaml && \
    conda clean -afy

SHELL ["bash", "-c"]
RUN echo "conda activate quake-env" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=test-env

# -----------------------------
# Entry point
# -----------------------------
ENTRYPOINT ["/bin/bash", "-ci", "conda activate quake-env && exec bash"]