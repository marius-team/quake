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


# Copy in your conda environment YAML
COPY environments/ubuntu-latest/conda.yaml /tmp/conda.yaml

# Create quake-env
RUN conda env create -f /tmp/conda.yaml && conda clean -afy

# -----------------------------
# Install PyTorch (CPU-only)
# -----------------------------
RUN conda run -n quake-env pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN echo "===== DEBUG: which conda =====" && which conda
RUN echo "===== DEBUG: conda info =====" && conda info
RUN echo "===== DEBUG: conda env list =====" && conda env list
RUN echo "===== DEBUG: quake-env check =====" && conda run -n quake-env python -c "import sys; print('OK in quake-env; python:', sys.executable)"

CMD ["/bin/bash"]