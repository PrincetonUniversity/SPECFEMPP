Build system requirements
=========================

This section covers compilation checks required so that SPECFEM++ is able to run
across all architectures.

.. note::

    This section is still under development.

Optional dependencies
----------------------

* **Trilinos** -- enables implicit (matrix-assembled) solvers. See
  :ref:`trilinos_configuration` for how to build SPECFEM++ against a shared
  Trilinos install. Trilinos must share the same Kokkos as SPECFEM++.

Intel GPU compilation
---------------------

SPECFEM++ uses SYCL for GPUs only; use the OpenMP or Serial backend on CPUs.
The program stops at startup if Kokkos selects a SYCL device that is not a GPU.

Load your Intel oneAPI compiler environment and use the shared ``release-sycl``
preset for portable SPIR-V with just-in-time (JIT) compilation:

.. code-block:: bash

   cmake --preset release-sycl
   cmake --build --preset release-sycl

The preset selects ``Kokkos_ARCH_INTEL_GEN=ON``. Kokkos supplies the target
compiler flags. For ahead-of-time (AOT) compilation, select the architecture
matching your GPU from the
`Kokkos Intel GPU configuration guide <https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#intel-gpus>`_.
``INTEL_GEN`` selects JIT; the specific Intel GPU architecture options select
AOT. Check that the option is available in the Kokkos version you are using.

For example, for a DG2 GPU, configure a separate build directory:

.. code-block:: bash

   cmake --preset release-sycl -B build/release-sycl-dg2 \
     -DKokkos_ARCH_INTEL_GEN=OFF -DKokkos_ARCH_INTEL_DG2=ON \
     -DCMAKE_INSTALL_PREFIX="$PWD/bin/release-sycl-dg2" \
     -DSPECFEM_BENCHMARKS_BUILD_DIR="$PWD/benchmarks/build/release-sycl-dg2"
   cmake --build build/release-sycl-dg2

Disable the inherited JIT architecture when selecting an AOT architecture.
Use the Intel offline compiler (``ocloc``) required by your compiler installation.
AOT binaries require a compatible target GPU and driver at runtime. If Kokkos
has no matching AOT architecture for your GPU, use ``INTEL_GEN`` for JIT.
Machine-specific architecture selections can be stored in the gitignored
``CMakeUserPresets.json`` by inheriting ``release-sycl`` and using separate
build, installation, and benchmark directories.
