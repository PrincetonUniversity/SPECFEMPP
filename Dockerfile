# Dockerfile to build serial version

FROM gcc:14.2.0

ENV SOURCE=/usr/local/specfempp/source
ENV BUILD=/usr/local/specfempp/build
ENV VTK_DIR=/usr/local/vtk/build


# Build and Install CMake
RUN echo "Installing CMake..." && \
    echo "====================" && \
    echo "" && \
    apt-get update && \
    apt-get install -y cmake && \
    echo "Done."
    # version=3.26 && \
    # build=5 && \
    # ## don't modify from here
    # mkdir ~/temp && \
    # cd ~/temp && \
    # wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz && \
    # tar -xzvf cmake-$version.$build.tar.gz && \
    # cd cmake-$version.$build/ && \
    # ./bootstrap --parallel=$(nproc) && \
    # make -j$(nproc) && \
    # make -j$(nproc) install && \
    # rm -rf ~/temp && \

RUN echo "cmake version:" && \
    cmake --version && \
    echo "Done."

RUN echo "Installing HDF5..." && \
    echo "==========================" && \
    echo "" && \
    apt-get install -y --no-install-recommends libhdf5-serial-dev && \
    echo "Done."

RUN echo "HDF5 version:" && \
    h5cc -showconfig && \
    echo "Done."

# Set up WORKDIR
WORKDIR /usr/local/vtk

RUN echo "Installing OSMesa..." && \
    echo "========================" && \
    echo "" && \
    apt-get install -y \
     mesa-utils \
     mesa-common-dev \
     libgl1-mesa-dev \
     libglu1-mesa-dev \
     libxt-dev \
     libxrender-dev \
     && \
    echo "Done."

RUN echo "Installing vtk..." && \
    echo "========================" && \
    echo "" && \
    wget https://www.vtk.org/files/release/9.4/VTK-9.4.2.tar.gz && \
    tar -xzf VTK-9.4.2.tar.gz && \
    mkdir ${VTK_DIR} && \
    cd ${VTK_DIR} && \
    cmake -S /usr/local/vtk/VTK-9.4.2 -B ${VTK_DIR} \
     -DCMAKE_BUILD_TYPE=Release \
     -DVTK_USE_X=ON \
     -DVTK_OPENGL_HAS_OSMESA=ON \
     -DVTK_USE_OSMESA=ON \
     -DVTK_DEFAULT_RENDER_WINDOW_HEADLESS=ON \
     -DVTK_RENDERING_BACKEND=OpenGL2 \
     -DVTK_GROUP_ENABLE_Rendering=WANT \
     -DBUILD_SHARED_LIBS=ON \
     -DVTK_MODULE_ENABLE_VTK_RenderingContextOpenGL2=YES \
     -DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2=YES \
     -DVTK_WRAP_PYTHON=OFF && \
    cmake --build ${VTK_DIR} -j2 && \
    echo "Done."


# Change dir to specfempp
WORKDIR /usr/local/specfempp

# Copy local source files to the container folder
COPY . ${SOURCE}

# VOLUME "/usr/local/specfempp/data"

# # Install SPECFEM++
RUN echo "Installing SPECFEM++..." && \
    echo "========================" && \
    echo "" && \
    cd ${SOURCE} && \
    rm -rf ${BUILD} && \
    cmake -S ${SOURCE} -B ${BUILD} -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=ON -D BUILD_EXAMPLES=ON && \
    cmake --build ${BUILD} -j2 && \
    echo "Done."

# Set environment variables
ENV PATH="${BUILD}/bin:${PATH}"

# Remove the source directory
# RUN echo "Cleaning up..." && \
#     echo "================" && \
#     echo "" && \
#     rm -rf ${SOURCE} && \
#     echo "Done."


ENV __GLX_VENDOR_LIBRARY_NAME=mesa
ENV LIBGL_ALWAYS_SOFTWARE=1

CMD ["/bin/bash"]
