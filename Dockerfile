FROM ubuntu:24.04

WORKDIR /

# Install required packages
RUN apt update && apt install -y git python3-pip cmake libblas-dev liblapack-dev libnuma-dev libgtest-dev

RUN pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . /quake

# Fix arm libgfortran name issue
RUN arch=$(uname -m) && \
if [ "$arch" = "aarch64" ]; then \
    echo "Running on ARM64"; \
    ln -s /usr/lib/aarch64-linux-gnu/libgfortran.so.5.0.0 /usr/lib/aarch64-linux-gnu/libgfortran-4435c6db.so.5.0.0 ; \
elif [ "$arch" = "x86_64" ]; then \
    echo "Running on AMD64"; \
else \
    echo "Unknown architecture: $arch"; \
fi

WORKDIR /quake

RUN mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
        -DQUAKE_ENABLE_GPU=OFF \
        -DQUAKE_USE_NUMA=OFF \
        -DQUAKE_USE_AVX512=OFF .. \
        -DBUILD_TESTS=ON .. \
        -DQUAKE_SET_ABI_MODE=OFF .. \
    && make bindings -j$(nproc) \
    && make quake_tests -j$(nproc)

RUN pip install --no-use-pep517 --break-system-packages .