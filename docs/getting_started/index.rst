
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
  - Supported by Kokkos: The compiler versions that are supported by Kokkos. We have not tested these versions for SPECFEM++, but in theory they should work.

.. list-table::
    :widths: 19 27 27 27
    :header-rows: 1
    :align: center

    * - Compiler
      - Recommended
      - Tested
      - Supported by Kokkos

    * * GNU
      * 8.5.0
      * 8.5.0, 13.2.1
      * 8.4.0, latest

    * * IntelLLVM
      * 2024.0.2
      * 2022.2.0, 2024.0.2
      * 2021.1.1, 2023.0.0

    * * NVCC
      * 12.6
      * 11.7, 12.6
      * 11.0, 11.6, 11.7

    * * Clang
      * Not Tested
      * Not Tested
      * 8.0.0, latest

    * * Apple Clang
      * 16.0.0 (MacOS Sequoia)
      * Not Tested
      * 8.0.0, latest

    * * NVC++
      * Not Tested
      * Not Tested
      * 22.3, 22.9

    * * ROCM
      * Not Tested
      * Not Tested
      * 5.2.0

    * * ARM/Clang
      * Not Tested
      * Not Tested
      * 20.1

Dependencies
~~~~~~~~~~~~

.. note::

    If any of the following dependencies are not found in your ``PATH``, the build process will download and install them automatically. This will increase the build time and does require an active internet connection.

* Kokkos: required
* Boost: required
* YAML-CPP: required
* HDF5: optional

Download SPECFEM++
------------------

Get the latest version of the package:


.. code-block:: bash

    git clone git@github.com:PrincetonUniversity/SPECFEMPP.git
    cd SPECFEMPP
    git submodule init
    git submodule update

Build recipes
-------------

SPECFEM++ inherits several architecure specific cmake configuration keywords from `Kokkos <https://kokkos.github.io/kokkos-core-wiki/keywords.html>`_. Below are the recommended build recipes for different architectures:

* CPU Serial version

.. code-block:: bash

    # cd into SPECFEM root directory
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=ON -D ENABLE_SIMD=ON -D Kokkos_ARCH_NATIVE=ON -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON -D Kokkos_ENABLE_ATOMICS_BYPASS=ON
    cmake --build build

* CPU OpenMP version

.. code-block:: bash

    # cd into SPECFEM root directory
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=ON -D ENABLE_SIMD=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ARCH_NATIVE=ON -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON
    cmake --build build

* CUDA version (needs cudatoolkit >= 11.7)

.. code-block:: bash

    # cd into SPECFEM root directory
    cmake3 -S . -B build -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_<architecture>=ON
    cmake --build build

.. note::

    Specify the architecture flag ``-D Kokkos_ARCH_<architecture>`` based on the GPU architecture you are using. For example, for NVIDIA Ampere architecture, use ``-D Kokkos_ARCH_AMPERE80=ON``. See `Kokkos documentation <https://kokkos.org/kokkos-core-wiki/keywords.html>`_ for more information.

Adding SPECFEM to PATH
----------------------

Finally, once compiled you could run SPECFEM++ from inside the build directory, by running the executible ``./specfem2d``. However, we recommend you add SPECFEM++ build directory to your ``PATH`` using

.. code-block:: bash

    export PATH=$(pwd)/build:$PATH

Running the solver
------------------

Lets run a simple example to test the installation. We will use the ``example\homogeneous-medium-flat-topography`` directory in the SPECFEM++ repository. The example directory contains a mesh of a homogeneous half-space with a single source and neumann boundary conditions.

.. note::

  A detailed description of the example can be found within :ref:`this cookbook <homogeneous_example>`

.. code-block:: bash

  cd examples/homogeneous-medium-flat-topography
  mkdir -p OUTPUT_FILES
  xmeshfem2D -p Par_File

This will generate the mesh files. Next, we will run the solver using

.. code-block:: bash

  mkdir -p OUTPUT_FILES/results
  specfem2d -p specfem_config.yaml

This will run the solver and generate synthetic seismograms at the receiver locations specified in ``STATIONS`` file.
