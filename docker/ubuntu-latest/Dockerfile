FROM ubuntu:20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and Python packages
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    /opt/miniconda/bin/conda install -y python=3.11 numpy pytorch cpuonly -c pytorch && \
    rm miniconda.sh

ENV PATH="/opt/miniconda/bin:${PATH}"