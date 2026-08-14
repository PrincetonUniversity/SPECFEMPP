# Dockerfile for the published SPECFEM++ image (CPU, serial)

# ubuntu:26.04 provides everything from apt: cmake 4.2, gcc 15, HDF5 1.14,
# and VTK 9.5. VTK >= 9.4 falls back to EGL/OSMesa at runtime when no X
# display is present, so wavefield PNG/JPG output works headless without
# xvfb. libegl-mesa0/libgl1-mesa-dri/libosmesa6 provide the software
# rendering backends VTK loads at runtime. mpi-default-dev is required
# because Ubuntu's VTK is MPI-enabled and its CMake config resolves MPI::MPI_C.
FROM ubuntu:26.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        cmake \
        git \
        ca-certificates \
        pkg-config \
        libhdf5-dev \
        hdf5-tools \
        zlib1g-dev \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-graph-dev \
        libvtk9-dev \
        mpi-default-dev \
        libegl1 \
        libegl-mesa0 \
        libgl1 \
        libgl1-mesa-dri \
        libosmesa6 \
        python3-venv \
        vim \
        emacs-nox \
    && rm -rf /var/lib/apt/lists/*

# snakemake in its own venv (PEP 668: system python is externally managed)
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir snakemake
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /usr/local/specfempp

ENV SOURCE=/usr/local/specfempp/source

COPY . ${SOURCE}

# CMAKE_POLICY_VERSION_MINIMUM: cmake 4.x rejects subprojects declaring
# cmake_minimum_required < 3.5 (yaml-cpp 0.8.0 declares 3.4).
#
# Portable no-SIMD build: safe on any CPU, default on PATH. Executables are
# installed to bin/<preset> by the organize_output target; the build tree is
# removed in the same layer to keep the image small.
RUN cd ${SOURCE} && \
    cmake --preset release-nosimd \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DSPECFEM_ENABLE_HDF5=ON \
        -DSPECFEM_ENABLE_VTK=ON \
        -DSPECFEM_BUILD_TESTS=OFF \
        -DSPECFEM_BUILD_BENCHMARKS=OFF && \
    cmake --build --preset release-nosimd -j "$(nproc)" && \
    rm -rf ${SOURCE}/build/release-nosimd

# SIMD build. Kokkos_ARCH_NATIVE would bake in the CI runner's CPU, so pin
# the x86-64-v3 baseline instead (AVX2+FMA; Intel Haswell 2013+, AMD Zen
# 2017+). amd64-only: the flag does not exist on arm64, where the no-SIMD
# build is already the portable baseline.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        cd ${SOURCE} && \
        cmake --preset release \
            -DKokkos_ARCH_NATIVE=OFF \
            -DCMAKE_CXX_FLAGS="-march=x86-64-v3" \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            -DSPECFEM_ENABLE_HDF5=ON \
            -DSPECFEM_ENABLE_VTK=ON \
            -DSPECFEM_BUILD_TESTS=OFF \
            -DSPECFEM_BUILD_BENCHMARKS=OFF && \
        cmake --build --preset release -j "$(nproc)" && \
        rm -rf ${SOURCE}/build/release ; \
    fi

# no-SIMD is the default; opt into the AVX2 build with
#   export PATH=/usr/local/specfempp/source/bin/release:$PATH
ENV PATH="${SOURCE}/bin/release-nosimd:${PATH}"

# OSMesa renders identically to native GL; the EGL/llvmpipe path has minor
# rasterization artifacts. Unset to let VTK pick (X -> EGL -> OSMesa).
ENV VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

# Smoke test: executables resolve their shared libraries and run.
RUN specfem --help > /dev/null && \
    if [ "$TARGETARCH" = "amd64" ]; then \
        ${SOURCE}/bin/release/specfem --help > /dev/null ; \
    fi

CMD ["/bin/bash"]
