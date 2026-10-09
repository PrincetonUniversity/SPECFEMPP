.. _tests:

Continuous Integration (CI)
===========================

As part of our Continuous Integration (CI) process, we run a series of automated
tests to ensure the stability and reliability of the codebase, as well as
automated documentation builds. The primary goal of these tests are to ensure
that the code can be compiled and run accurately on various supported platforms,
and that new changes do not introduce regressions or break existing
functionality. We use a combination of Github Actions and Jenkins to run these
tests, the details of which are described below.

Read the Docs
-------------

.. |rdt_proj| replace:: ``specfem2d_kokkos``

We host documentation on Read the Docs under the project name |rdt_proj|_ . The
documentation is built using the configuration file
:repo-file:`.readthedocs.yml`. The documentation is automatically built on every
push to the repository and on every pull request, to easily identify
documentation issues.

.. _rdt_proj: https://app.readthedocs.org/projects/specfem2d-kokkos/?utm_source=specfem2d-kokkos&utm_content=flyout

Github Actions
--------------

Partial compilation checks and unit tests
+++++++++++++++++++++++++++++++++++++++++

On Github actions, there are two types of tests that are run on every pull
request or push to the repository:

1. **CPU Compilation checks**: The goal is to ensure current push doesn't break
   the compilation. These tests would run on forks of this repository.
   Ultimately, the hope is that end developer commits their changes to local
   fork at regualar intervals which would reduce compilation errors during
   development process.
   The cpu compilation checks are defined in :repo-file:`.github/workflows/compilation.yml`

2. **CPU unit tests**: The tests are run in a serial mode using GNU compilers.
   The goal is to ensure current push doesn't break the unit tests. These tests
   would run on forks of this repository. Ultimately, the hope is that end
   developer commits their changes to local fork at regular intervals which
   would reduce unit test errors during development process.
   The cpu unit tests are defined in :repo-file:`.github/workflows/unittests.yml`

Docker
++++++

We use Docker to give users the ability to easily build and run the code with
extensive configuration options. The Docker configuration file is in the root
directory of the repository as :repo-file:`Dockerfile`. We host two types builds through
github release builds that have a version number and builds based on the
``devel`` branch. The github workflow configuration file is in
:repo-file:`.github/workflows/docker.yml`.


Jenkins - Complete compilation and unit tests
---------------------------------------------

We also have a Jenkins server that runs more exhaustive tests on every
maintainer pull request to the repository and on request from contributors from
forks. If an external contributor would like to run these tests on their pull
request, then a maintainer will have to comment ``please this this`` or ``retest
this please`` on the pull request, to launch these tests. Pull requests can only
be merged if these tests pass (and maintainers approve the pull request).

We run a matrix of compilation and unit tests on various supported compilers and
and options for both CPU and GPU, which are summarized below.

CPU
+++

**Gnu Compiler Collection (GCC)**

This is defined in :repo-file:`.jenkins/gnu_compiler_checks.gvy`

- GCC Versions: 11.5.0, 14.2.1
- Compilation modes: Serial, OpenMP
- SIMD options: On, Off

The resulting test combinations are:

.. list-table::
   :header-rows: 1
   :widths: 20 25 20

   * - GCC Version
     - Compilation Mode
     - SIMD Option
   * - 11.5.0
     - Serial
     - On
   * - 11.5.0
     - Serial
     - Off
   * - 11.5.0
     - OpenMP
     - On
   * - 11.5.0
     - OpenMP
     - Off
   * - 14.2.1
     - Serial
     - On
   * - 14.2.1
     - Serial
     - Off
   * - 14.2.1
     - OpenMP
     - On
   * - 14.2.1
     - OpenMP
     - Off

**Intel OneAPI Compiler**

This is defined in :repo-file:`.jenkins/intel_compiler_checks.gvy`

- Intel compiler versions: 2024.2.0
- Compilation modes: Serial, OpenMP
- SIMD options: On, Off

The resulting test combinations are:

.. list-table::
   :header-rows: 1
   :widths: 20 25 20

   * - Intel Version
     - Compilation Mode
     - SIMD Option
   * - 2024.2.0
     - Serial
     - On
   * - 2024.2.0
     - Serial
     - Off
   * - 2024.2.0
     - OpenMP
     - On
   * - 2024.2.0
     - OpenMP
     - Off

GPU
+++

**NVIDIA CUDA Compiler (NVCC)**

This is defined in :repo-file:`.jenkins/cuda_compiler_checks.gvy`. Currently the only
architecture that is tested is NVIDIA Ampere (A100).

- CPU Compiler: GNU 11.5.0
- CUDA: :repo-file:`cudatoolkit/11.8`, :repo-file:`cudatoolkit/12.8`
- Compilation modes: Serial, OpenMP
- SIMD options: On, Off

The resulting test combinations are:

.. list-table::
   :header-rows: 1
   :widths: 20 25 20

   * - CUDA Version
     - Compilation Mode
     - SIMD Option
   * - 11.8
     - Serial
     - On
   * - 11.8
     - Serial
     - Off
   * - 11.8
     - OpenMP
     - On
   * - 11.8
     - OpenMP
     - Off
   * - 12.8
     - Serial
     - On
   * - 12.8
     - Serial
     - Off
   * - 12.8
     - OpenMP
     - On
   * - 12.8
     - OpenMP
     - Off

Trilinos implicit solvers
+++++++++++++++++++++++++

This is defined in :repo-file:`.jenkins/trilinos_compiler_checks.gvy`. It is the only
job that builds with ``SPECFEM_ENABLE_TRILINOS=ON``, and therefore the only one that
actually compiles and runs the implicit Newmark solver and the
:doc:`linear system </sections/api/specfem/linear_system/index>` assemblers. In every
other job that code is behind ``#ifdef SPECFEM_ENABLE_TRILINOS`` and its tests
``GTEST_SKIP()``.

The build configuration is almost entirely pinned by the Trilinos install it links
against, so this job has only one real axis:

- Trilinos module: ``trilinos/17.1.1-cpu-native-nompi``
- Compiler: GNU 14.2.1 (``gcc-toolset/14``) -- must match the compiler Trilinos was
  built with
- Host space: Serial only -- Kokkos comes from the Trilinos module via
  ``find_package``, so it cannot be rebuilt with a different backend here
- Precision: single -- the cluster Trilinos is built ``Tpetra_INST_FLOAT`` only
- SIMD options: On, Off

The resulting test combinations are:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 15

   * - Trilinos Version
     - Compilation Mode
     - Precision
     - SIMD Option
   * - 17.1.1 (CPU, no MPI)
     - Serial
     - Single
     - On
   * - 17.1.1 (CPU, no MPI)
     - Serial
     - Single
     - Off

Selecting the Trilinos tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tests that require a Trilinos build carry the ``TRILINOS`` `CTest label
<https://cmake.org/cmake/help/latest/prop_test/LABELS.html>`_, set through
``specfem_add_test(... LABELS TRILINOS)``. The job selects them with:

.. code-block:: bash

    ctest -L TRILINOS --output-on-failure --no-tests=error

Locally, the inverse is useful: the implicit-solver integration test runs full explicit
reference simulations and takes several minutes, so

.. code-block:: bash

    ctest -LE TRILINOS

skips everything that needs Trilinos, including that test.

Note that ``element_stiffness_tests`` is deliberately **not** labelled. The element
stiffness kernel is Trilinos-free by design, so it already runs for real in every other
job and should not be excluded by ``-LE TRILINOS``.

.. note::

    ``specfem_add_test`` accepts only a **single** label in practice.
    ``gtest_discover_tests()`` flattens a list-valued test property when it generates its
    discovery file, so ``LABELS a b`` reaches ``set_tests_properties()`` as two key/value
    pairs: ``LABELS`` binds to ``a`` and ``b`` is treated as a property name of its own.
    The second label is dropped with no CMake or CTest diagnostic.

Results are published to Jenkins as JUnit XML (``ctest --output-junit``) and archived as
build artifacts, so individual test failures are visible in the Jenkins UI rather than
only in the console log.
