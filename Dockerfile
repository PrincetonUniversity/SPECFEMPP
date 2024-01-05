# Dockerfile to build serial version

FROM gcc:9.5.0

# Set up WORKDIR
WORKDIR /usr/local/specfempp

COPY . .

# Install CMake
RUN echo "Installing CMake..." && \
    echo "====================" && \
    echo "" && \
    apt-get update && \
    apt-get install -y cmake && \
    echo "Done."

RUN echo "cmake version:" && \
    cmake --version && \
    echo "Done."

RUN echo $(ls -ltra .)

## TODO: Install boost from a tarball

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
