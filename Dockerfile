# Dockerfile to build serial version

# Kokkos 5.2 requires GCC >= 10.4 for C++20
FROM gcc:13


# Install vim editor
RUN apt-get update && apt-get install -y vim wget build-essential git

# Install Emacs
RUN apt-get update && apt-get install -y emacs

# Install snakemake (--break-system-packages needed on Debian bookworm, PEP 668)
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install --break-system-packages snakemake

# Install CMake (prebuilt binary)
RUN echo "Installing CMake..." && \
    echo "====================" && \
    echo "" && \
    version=3.26 && \
    build=5 && \
    arch=$(uname -m) && \
    mkdir ~/temp && \
    cd ~/temp && \
    wget https://github.com/Kitware/CMake/releases/download/v$version.$build/cmake-$version.$build-linux-$arch.sh && \
    sh cmake-$version.$build-linux-$arch.sh --skip-license --prefix=/usr/local && \
    rm -rf ~/temp && \
    echo "Done."

RUN echo "cmake version:" && \
    cmake --version && \
    echo "Done."

# Install HDF5
RUN echo "Installing HDF5..." && \
    echo "==================" && \
    echo "" && \
    apt-get update && \
    apt-get install -y \
        libhdf5-dev \
        libhdf5-serial-dev \
        hdf5-tools \
        pkg-config \
        zlib1g-dev && \
    echo "Done."

# Set HDF5 environment variables for CMake discovery
ENV HDF5_ROOT=/usr
ENV HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial
ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig

# Verify HDF5 installation
RUN echo "HDF5 version:" && \
    h5dump --version && \
    echo "HDF5 libraries location:" && \
    find /usr -name "libhdf5*" -type f 2>/dev/null | head -10 && \
    echo "Done."

# Set up WORKDIR
WORKDIR /usr/local/specfempp

ENV SOURCE=/usr/local/specfempp/source
ENV BUILD=/usr/local/specfempp/source/build

COPY . ${SOURCE}
# Install SPECFEM++
RUN echo "Installing SPECFEM++..." && \
    echo "========================" && \
    echo "" && \
    cd ${SOURCE} && \
    rm -rf ${BUILD} && \
    cmake --preset release-nosimd -DSPECFEM_ENABLE_HDF5=ON -DSPECFEM_ENABLE_VTK=OFF && \
    cmake --build --preset release-nosimd && \
    echo "Done."

# Set environment variables
ENV PATH="${SOURCE}/bin/release-nosimd:${PATH}"

CMD ["/bin/bash"]
