
Getting Started
===============

This guide is intended to help new users get started with running SPECFEM++ simulations.

Requirements
------------

Build system
~~~~~~~~~~~~

* CMake >= ``3.16``: required
* CMake >= ``3.21.1`` for NVC++

Compiler Versions
~~~~~~~~~~~~~~~~~

The following table lists the versions of compilers that are supported by SPECFEM++:
  - Recommended: The compiler versions that are tested for performance and stability.
  - Tested: The compiler versions that are tested for stability.
  - Kokkos Tested: The compiler versions that are tested by Kokkos. We have not tested these versions for SPECFEM++, but in theory they should work.
  - Minimum Version: The minimum compiler versions required to build SPECFEM++.

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1
    :align: center

    * - Compiler
      - Recommended
      - Tested
      - Kokkos Tested
      - Minimum Version

    * * GNU
      * 11.5.0
      * 11.5.0, 14.2.1
      * 8.4.0, latest
      * 8.2.0

    * * IntelLLVM
      * 2024.0.2
      * 2024.2
      * 2024.2
      * 2021.1.1

    * * NVCC
      * 12.8
      * 11.8, 12.8
      * 11.0, 11.6, 11.7
      * 11.0

    * * Apple Clang
      * 16.0.0 (MacOS Sequoia)
      * Not Tested
      * latest
      * 8.0.0

    * * NVC++
      * Not Tested
      * Not Tested
      * 22.9
      * 22.3

    * * ROCM
      * Not Tested
      * Not Tested
      * 5.2.0
      * 5.2.0

    * * ARM/Clang
      * Not Tested
      * Not Tested
      * 20.1
      * 20.1


Dependencies
~~~~~~~~~~~~


Required (automatically installed if not found)
+++++++++++++++++++++++++++++++++++++++++++++++

If any of the following required dependencies are not found in
your ``PATH``, the build process will download and install them automatically.
This will increase the build time and does require an active internet
connection.

* **Kokkos** -- a C++ library for performance portability across many-core
  architectures. It is used for parallel programming in SPECFEM++.
* **Boost** >= ``1.85.0`` -- a collection of C++ libraries that provide support for
  file I/O.
* **YAML-CPP** -- a YAML parser and emitter in C++. It is used for reading writing
  configuration files in SPECFEM++.

Optional
++++++++

* **HDF5** can be used for reading and writing wavefield data. Specify custom
  ``hdf5`` builds using ``-D HDF5_DIR=/path/to/hdf5`` and add the libary path to
  the ``LD_LIBRARY_PATH`` environment variable:

  .. code-block:: bash

      export LD_LIBRARY_PATH=/path/to/hdf5/lib[64]:$LD_LIBRARY_PATH

* **ZLIB** can be used for reading and writing data in npz (zipped numpy arrays )format. Specify custom
  ``zlib`` builds using ``-D ZLIB_ROOT=/path/to/zlib`` and add the libary path to
  the ``LD_LIBRARY_PATH`` environment variable:

  .. code-block:: bash

      export LD_LIBRARY_PATH=/path/to/zlib/lib[64]:$LD_LIBRARY_PATH

* **VTK** can be used visualization of the wavefield in 2D. Specify a custom
  ``vtk`` build using ``-D VTK_DIR=/path/to/vtk`` and add the libary path to
  the ``LD_LIBRARY_PATH`` environment variable:

  .. code-block:: bash

      export LD_LIBRARY_PATH=/path/to/vtk/lib[64]:$LD_LIBRARY_PATH

.. note::

    If you are using ``module load`` to load the dependencies, the
    ``LD_LIBRARY_PATH`` and root directories of the optional build are often
    automatically set.


Download SPECFEM++
------------------

Get the latest version of the package:


.. code-block:: bash

    git clone git@github.com:PrincetonUniversity/SPECFEMPP.git
    cd SPECFEMPP

Build recipes
-------------

SPECFEM++ inherits several architecure specific cmake configuration keywords
from `Kokkos <https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html>`_.
Below are the recommended build recipes for different architectures:

* CPU Serial version

.. code-block:: bash

    # cd into SPECFEM root directory
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D SPECFEM_BUILD_TESTS=ON -D SPECFEM_ENABLE_SIMD=ON -D Kokkos_ARCH_NATIVE=ON -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON -D Kokkos_ENABLE_ATOMICS_BYPASS=ON
    cmake --build build

* CPU OpenMP version

.. code-block:: bash

    # cd into SPECFEM root directory
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D SPECFEM_BUILD_TESTS=ON -D SPECFEM_ENABLE_SIMD=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ARCH_NATIVE=ON -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON
    cmake --build build

* CUDA version (needs cudatoolkit >= 11.7)

.. code-block:: bash

    # cd into SPECFEM root directory
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D SPECFEM_BUILD_TESTS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_<architecture>=ON
    cmake --build build

.. note::

    Specify the architecture flag ``-D Kokkos_ARCH_<architecture>`` based on
    the GPU architecture you are using. For example, for NVIDIA Ampere
    architecture, use ``-D Kokkos_ARCH_AMPERE80=ON``. See
    `Kokkos documentation <https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#gpu-architectures>`_
    for more information.

.. note::
    To speedup compilation, you can enable parallel compilation by adding the
    ``-j <number_of_jobs>`` flag to the ``cmake --build build`` command. For
    example, to use 4 parallel jobs, you can run:

    .. code-block:: bash

        cmake --build build -j 4

.. note::
    When you have the need to switch between different build configurations
    (e.g., from CPU to CUDA), it is recommended to use :doc:`CMake presets </sections/getting_started/presets>` to
    manage the build configurations.

Adding SPECFEM to PATH
----------------------

Finally, once compiled you could run SPECFEM++ from inside the executable
directory ``./bin``, by running the executable ``./specfem``. However, we
recommend you add SPECFEM++ build directory to your ``PATH`` using

.. code-block:: bash

    export PATH=$(pwd)/bin:$PATH

Running the solver
------------------

Lets run a simple example to test the installation. We will use the
``example/homogeneous-medium-flat-topography`` directory in the SPECFEM++
repository. The example directory contains a mesh of a homogeneous half-space
with a single source and neumann boundary conditions.

.. note::

  A detailed description of the example can be found within
  :ref:`this cookbook <homogeneous_example>`

.. code-block:: bash

  cd examples/dim2/homogeneous-elastic
  mkdir -p OUTPUT_FILES
  xmeshfem2D -p Par_File

This will generate the mesh files. Next, we will run the solver using

.. code-block:: bash

  mkdir -p OUTPUT_FILES/results
  specfem 2d -p specfem_config.yaml

This will run the solver and generate synthetic seismograms at the receiver
locations specified in ``STATIONS`` file.
