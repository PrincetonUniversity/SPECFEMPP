
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

.. note::

    The following compilers are supported and tested by Kokkos. In theory, SPECFEM++ should work with any of these compiler versions. However we have not tested all of them and cannot guarantee the same. If you have issues compiling with a compiler versions listed below, please create an `issue on GitHub <https://github.com/PrincetonUniversity/specfem2d_kokkos/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=>`_.

.. list-table::
    :widths: 30 35 35
    :header-rows: 1
    :align: center

    * - Compiler
      - Minimum version
      - Primary tested versions

    * * GCC
      * 8.2.0
      * 8.4.0, latest

    * * Clang
      * 8.0.0
      * 8.0.0, latest

    * * IntelLLVM
      * 2021.1.1
      * 2023.0.0

    * * NVCC
      * 11.0
      * 11.0, 11.6, 11.7

    * * NVC++
      * 22.3
      * 22.9

    * * ROCM
      * 5.2.0
      * 5.2.0

    * * ARM/Clang
      * 20.1
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
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=ON -D ENABLE_SIMD=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ARCH_NATIVE=ON -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON -D Kokkos_ENABLE_ATOMICS_BYPASS=ON
    cmake --build build

* CUDA version (needs cudatoolkit >= 11.7)

.. code-block:: bash

    # cd into SPECFEM root directory
    cmake3 -S . -B build_gpu -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_<architecture>=ON -D BUILD_EXAMPLES=ON -D CMAKE_VERBOSE_MAKEFILE=ON
    cmake --build build

.. note::

    Specify the architecture flag ``-D Kokkos_ARCH_<architecture>`` based on the GPU architecture you are using. For example, for NVIDIA Ampere architecture, use ``-D Kokkos_ARCH_AMPERE80=ON``.

Adding SPECFEM to PATH
----------------------

Finally, once compiled you could run SPECFEM++ from inside the build directory, by running the executible ``./specfem``. However, we recommend you add SPECFEM++ build directory to your ``PATH`` using

.. code-block:: bash

    export PATH=${PATH}:<location to build directory>
