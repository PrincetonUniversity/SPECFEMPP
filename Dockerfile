# Dockerfile to build serial version

FROM gcc/9.5.0

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

# Install Boost 1.73.0
RUN echo "Installing Boost 1.73.0..." && \
    echo "==========================" && \
    echo "" && \
    apt-get install -y libboost1.73-dev && \
    echo "Done."

RUN echo "Boost version:" && \
    echo $(dpkg -s libboost1.73-dev | grep Version) && \
    echo "Done."

# Install SPECFEM++
RUN echo "Installing SPECFEM++..." && \
    echo "========================" && \
    echo "" && \
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release && \
    cmake --build build && \
    echo "Done."

# Set environment variables
ENV PATH="/usr/local/specfempp/build:${PATH}"

CMD ["/bin/bash"]
