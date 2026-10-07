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


Configure and build
-------------------

The ``release-trilinos`` :doc:`preset </sections/getting_started/presets>`
inherits the standard ``release`` presets and only set
``SPECFEM_ENABLE_TRILINOS=ON``. ``Trilinos_ROOT`` and ``Kokkos_ROOT`` are taken from
the loaded module's environment, so the presets stay machine-independent.

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
