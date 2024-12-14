FROM ubuntu:20.04

# -----------------------------
# Set up environment variables
# -----------------------------
ENV CONDA_DIR=/opt/miniconda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Miniconda
# -----------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# -----------------------------
# Set up and activate conda environment
# -----------------------------
COPY conda.yaml /tmp/conda.yaml
RUN conda env create -f /tmp/conda.yaml && \
    conda clean -afy \

SHELL ["bash", "-c"]
RUN echo "conda activate test-env" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=test-env

# -----------------------------
# Entry point
# -----------------------------
ENTRYPOINT ["/bin/bash", "-ci", "conda activate test-env && exec bash"]