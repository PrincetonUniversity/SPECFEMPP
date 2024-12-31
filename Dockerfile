# Dockerfile to build serial version

FROM gcc:9.5.0

# Set up WORKDIR
WORKDIR /usr/local/specfempp

COPY . .

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
    make install && \
    echo "Done."

RUN echo "cmake version:" && \
    cmake --version && \
    echo "Done."

# Install Boost
RUN echo "Installing Boost..." && \
    echo "====================" && \
    echo "" && \
    wget https://archives.boost.io/release/1.73.0/source/boost_1_73_0.tar.bz2 && \
    tar --bzip2 -xf boost_1_73_0.tar.bz2 && \
    cd boost_1_73_0 && \
    ./bootstrap.sh --prefix=/usr/local/boost_1_73_0 && \
    ./b2 install

# Install SPECFEM++
RUN echo "Installing SPECFEM++..." && \
    echo "========================" && \
    echo "" && \
    git submodule init && git submodule update && \
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=ON -D BUILD_EXAMPLES=ON && \
    cmake --build build && \
    echo "Done."

# Set environment variables
ENV PATH="/usr/local/specfempp/build:${PATH}"

CMD ["/bin/bash"]
