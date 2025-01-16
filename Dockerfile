# Dockerfile to build serial version

FROM gcc:9.5.0

# Set up WORKDIR
WORKDIR /usr/local/specfempp

ENV SOURCE=/usr/local/specfempp/source
ENV BUILD=/usr/local/specfempp/build

COPY . ${SOURCE}

# Build and Install CMake
RUN echo "Installing CMake..." && \
    echo "====================" && \
    echo "" && \
    apt-get update && \
    apt purge cmake && \
    version=3.26 && \
    build=5 && \
    ## don't modify from here
    mkdir ~/temp && \
    cd ~/temp && \
    wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz && \
    tar -xzvf cmake-$version.$build.tar.gz && \
    cd cmake-$version.$build/ && \
    ./bootstrap --parallel=$(nproc) && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf ~/temp && \
    echo "Done."

RUN echo "cmake version:" && \
    cmake --version && \
    echo "Done."

# Install SPECFEM++
RUN echo "Installing SPECFEM++..." && \
    echo "========================" && \
    echo "" && \
    cd ${SOURCE} && \
    git submodule init && git submodule update && \
    rm -rf ${BUILD} && \
    cmake -S ${SOURCE} -B ${BUILD} -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=ON -D BUILD_EXAMPLES=ON && \
    cmake --build ${BUILD} && \
    rm -rf ${SOURCE} && \
    echo "Done."

# Set environment variables
ENV PATH="${BUILD}/bin:${PATH}"

CMD ["/bin/bash"]
