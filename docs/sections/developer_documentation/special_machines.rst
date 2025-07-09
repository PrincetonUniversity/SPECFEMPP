
Compiling and Running on Special Machines
=========================================


Frontier (Oak Ridge National Laboratory)
----------------------------------------


At Oak Ridge National Laboratory, the Frontier supercomputer uses programming
environments and modules to manage software dependencies. Below are guides
to both CPU and GPU compilation.

CPU
+++

.. code:: bash

    # Load the necessary modules for Frontier
    module load PrgEnv-gnu # -> sets cc, CC, ftn to GNU compilers
    module load craype-x86-milan # Target AMD EPYC 3rd gen
    module load boost/1.86.0
    module load kokkos/4.4.01-omp

    # Resetting aliases for CMake compilation
    export CC=$(which cc)
    export CXX=$(which CC)
    export FC=$(which ftn)

    # Configure
    cmake --preset release

    # Build
    cmake --build --preset release -j

    # Test
    ctest --test-dir build/release/tests/unit-tests/


GPU
+++

.. code:: bash

    # Load the necessary modules for Frontier
    module load PrgEnv-amd # -> sets cc, CC, ftn to GNU compilers
    module load craype-x86-milan # Target AMD EPYC 3rd gen
    module load boost/1.86.0
    module load rocm/6.2.4              # Load ROCm for GPU support
    module load kokkos/4.4.01-gpu       # Needs to be loaded after rocm
    module load craype-accel-amd-gfx90a # Target AMD MI250x GPUs

    # Resetting aliases for CMake compilation
    export CC=$(which cc)
    export CXX=$(which hipcc)
    export FC=$(which ftn)

    # Configure
    cmake --preset release-frontier

    # Build
    cmake --build --preset release-frontier -j

    # Test
    ctest --test-dir build/release-frontier/tests/unit-tests/
