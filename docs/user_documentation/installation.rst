Installation
###############

Downloading SPECFEM
===================

Get the latest version of the package:

.. code-block:: bash

    <Get latest version of the software using git clone>
    git submodule init
    git submodule update

Compilation
============

Configure the package using Cmake configuration keywords before building using cmake. SPECFEM inherits several architecure specific keywords from `Kokkos <https://kokkos.github.io/kokkos-core-wiki/keywords.html>`_ Cmakelists. Several examples for compiling for different architectures are shown below

* CPU Serial version

.. code-block:: bash

    # cd into SPECFEM root directory
    cmake3 -S . -B build # creates a build directory to build/ inside SPECFEM root
    cmake3 --build build

* CPU OpenMP version

.. code-block:: bash

    cmake3 -S . -B build -DKokkos_ENABLE_OPENMP=ON
    cmake3 --build build

* OpenMP / CUDA enabled

.. code-block:: bash

    cmake3 -S . -B build -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON
    cmake3 --build build

Adding SPECFEM to PATH
======================

Finally, once compiled you could run specfem from inside the build directory, by running the executible ``./specfem``. However, we recommend you add SPECFEM build directory to your ``PATH`` using

.. code-block:: bash

    export PATH=${PATH}:<location to build directory>
