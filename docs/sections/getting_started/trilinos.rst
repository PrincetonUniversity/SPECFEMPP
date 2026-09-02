.. _trilinos_configuration:

Configuring SPECFEM++ with Trilinos
===================================

SPECFEM++ can optionally be built against `Trilinos
<https://trilinos.github.io/>`_ to assemble the global system matrix and solve it
with implicit linear solvers (as an alternative to the default matrix-free explicit
time marching). This support is **off by default** and is enabled with the
``SPECFEM_ENABLE_TRILINOS`` CMake option.

When enabled, SPECFEM++ links the following Trilinos packages:

* **Tpetra** -- distributed sparse matrices and vectors (Kokkos-based),
* **Belos** -- iterative linear solvers (e.g. GMRES, CG),
* **Ifpack2** -- incomplete-factorization / relaxation preconditioners,
* **MueLu** -- algebraic multigrid preconditioners,
* **Amesos2** -- sparse direct solvers.

.. important::

    Trilinos and SPECFEM++ **must share the same Kokkos** -- identical version,
    architecture, execution space, CUDA relocatable-device-code setting, and C++
    standard. The supported workflow installs a Trilinos that bundles Kokkos and
    points SPECFEM++ at that same Kokkos. Mixing two different Kokkos builds will
    fail to compile or link.

Trilinos is **not** downloaded or built by SPECFEM++; you must provide an install
and make it discoverable through ``Trilinos_ROOT`` (and the matching ``Kokkos_ROOT``).
On TROMP machines this is handled by the shared ``trilinos`` modules.

Using the TROMP Trilinos modules
--------------------------------

Pre-built Trilinos installs are provided as Lmod modules. Loading one exports both
``Trilinos_ROOT`` and ``Kokkos_ROOT`` so that SPECFEM++ uses the bundled Kokkos
(via ``find_package(Kokkos)`` instead of fetching its own copy):

.. list-table:: Available ``trilinos`` module variants
    :widths: 40 15 20 10
    :header-rows: 1
    :align: center

    * - Module
      - Backend
      - Kokkos arch
      - MPI
    * - ``trilinos/17.1.1-cpu-native-nompi``
      - Serial
      - ``NATIVE``
      - no
    * - ``trilinos/16.1.0-cuda-ampere80-mpi``
      - CUDA
      - ``AMPERE80`` + ``NATIVE``
      - yes
    * - ``trilinos/16.1.0-cuda-ampere80-nompi``
      - CUDA
      - ``AMPERE80`` + ``NATIVE``
      - no
    * - ``trilinos/16.1.0-cpu-native-mpi``
      - Serial
      - ``NATIVE``
      - yes
    * - ``trilinos/16.1.0-cpu-native-nompi``
      - Serial
      - ``NATIVE``
      - no

Choose the variant that matches how you build SPECFEM++ (GPU vs CPU, with or
without MPI).

.. warning::

    Only ``trilinos/17.1.1-cpu-native-nompi`` is currently installed. The four
    ``trilinos/16.1.0-*`` modulefiles still load, but the prefixes they point at under
    ``/home/TROMP/source/Trilinos/install/`` have been removed, so ``find_package(Trilinos)``
    will fail. Until they are rebuilt, only the ``release-trilinos`` preset can be
    configured -- ``release-trilinos-mpi``, ``release-cuda-trilinos`` and
    ``release-cuda-trilinos-mpi`` have no backing install.

    This install is also built ``Tpetra_INST_FLOAT`` only, so SPECFEM++ must be built in
    single precision (``SPECFEM_ENABLE_DOUBLE_PRECISION=OFF``, the default). A double
    build finds no Belos/Ifpack2 instantiation for ``type_real`` and fails to link.

.. important::

    **Load the compiler module after the Trilinos module.** The ``trilinos`` modulefiles
    auto-load ``openblas/0.3.x``, whose only effect is to prepend ``/usr/lib64`` to
    ``LD_LIBRARY_PATH``. That places it ahead of the ``gcc-toolset-14`` tree, so the
    toolset linker picks up the system ``libctf.so.0`` and every link fails with
    ``version `LIBCTF_1.1' not found``. CMake reports this as
    ``The C++ compiler ... is not able to compile a simple test program``, which points at
    the compiler rather than at the module environment.

    Loading ``gcc-toolset/14`` last restores the correct ordering:

    .. code-block:: bash

        module load trilinos/17.1.1-cpu-native-nompi
        module load gcc-toolset/14

    Equivalently, if the compiler must be loaded first, fix the path explicitly:

    .. code-block:: bash

        export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-14/root/usr/lib64:$LD_LIBRARY_PATH

Configure and build
-------------------

The ``release-trilinos`` and ``release-cuda-trilinos`` :doc:`presets </sections/getting_started/presets>`
inherit the standard ``release`` / ``release-cuda`` presets and only set
``SPECFEM_ENABLE_TRILINOS=ON``. ``Trilinos_ROOT`` and ``Kokkos_ROOT`` are taken from
the loaded module's environment, so the presets stay machine-independent.

GPU (NVIDIA Ampere / A100):

.. code-block:: bash

    module load trilinos/16.1.0-cuda-ampere80-mpi
    cmake --preset release-cuda-trilinos
    cmake --build build/release-cuda-trilinos

CPU (the only variant currently installed):

.. code-block:: bash

    module load trilinos/17.1.1-cpu-native-nompi
    module load gcc-toolset/14
    cmake --preset release-trilinos
    cmake --build build/release-trilinos

Equivalently, without presets, pass the option directly (with a module loaded, or
by pointing at an install prefix explicitly):

.. code-block:: bash

    cmake -S . -B build \
        -D SPECFEM_ENABLE_TRILINOS=ON \
        -D Trilinos_ROOT=/path/to/trilinos/install \
        -D Kokkos_ROOT=/path/to/trilinos/install
    cmake --build build

A successful configuration prints ``Found Trilinos <version>`` and reuses the
module's Kokkos (no second Kokkos is downloaded).

.. note::

    The Trilinos installs and their build scripts live in the TROMP shared tree at
    ``/home/TROMP/source/Trilinos/scripts/`` (build scripts and module sources) and
    are installed under ``/home/TROMP/source/Trilinos/install/``. Maintainers should
    use those scripts to rebuild or add variants rather than duplicating them here.
