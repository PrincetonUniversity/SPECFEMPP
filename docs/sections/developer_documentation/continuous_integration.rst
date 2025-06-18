.. _tests:

Continuous Integration (CI)
===========================

The following tests are run as part of continuous integration (CI) process. The goal of these tests are to ensure that the code can be compiled and run accurately on various supported platforms.

1. **Partial compilation checks (github actions)**: These tests are run on every push to the repository. This check tests if the code can be copiled using GNU compilers in serial mode. The goal is to ensure current push doesn't break the compilation. These tests would run on forks of this repository. Ultimately, the hope is that end developer commits their changes to local fork at regualar intervals which would reduce compilation errors during development process.

2. **Partial unit tests (github actions)**: These tests are run on every push to the repository. The tests are run in a serial mode using GNU compilers. The goal is to ensure current push doesn't break the unit tests. These tests would run on forks of this repository. Ulitmately, the hope is that end developer commits their changes to local fork at regualar intervals which would reduce unit test errors during development process.

3. **Complete compilation checks (Jenkins)**: These tests are run on every pull request to the repository. The tests ensure the code can be compiled on various supported platforms. If you're are first time contributor, then an admin would have to approve your request to run these tests. The pull request would be merged only if these tests pass.

   list of tested compilers:

   - CPU: GNU 11.5.0, GNU 14.2.1, Intel 2024.2.0
   - CUDA: cudatoolkit/11.7, cudatoolkit/12.8

   list of tested GPU architectures:

   - NVIDIA Ampere: A100

   Currently, GPU compilation is only checked using GNU compilers. The following matrix shows the list of compilers and GPU architectures that are tested:

   .. rst-class:: center-table

   +------------------------------+--------------------+---------------------+
   |                              |        NONE        | NVIDIA Ampere: A100 |
   +==============================+====================+=====================+
   | GNU 8.5.0 (serial mode)      |         ✓          |          ✓          |
   +------------------------------+--------------------+---------------------+
   | Intel 2020.2.0 (serial mode) |         ✓          |          ✘          |
   +------------------------------+--------------------+---------------------+
   | GNU 8.5.0 (OpenMP)           |         ✓          |          ✓          |
   +------------------------------+--------------------+---------------------+
   | Intel 2020.2.0 (OpenMP)      |         ✓          |          ✘          |
   +------------------------------+--------------------+---------------------+

4. **Complete unit tests (Jenkins)**: These tests are run on every pull request to the repository. The tests ensure the code runs accurately on various supported platforms. If you're are first time contributor, then an admin would have to approve your request to run these tests. The pull request would be merged only if these tests pass.

   list of tested compilers:

   - CPU: GNU 11.5.0, GNU 14.2.1, Intel 2024.2.0
   - CUDA: cudatoolkit/11.7, cudatoolkit/12.8

   list of tested GPU architectures:

   - NVIDIA Ampere: A100

   Currently, GPU compilation is only checked using GNU compilers. The following matrix shows the list of compilers and GPU architectures that are tested:

   .. rst-class:: center-table

   +------------------------------+--------------------+---------------------+
   |                              |        NONE        | NVIDIA Ampere: A100 |
   +==============================+====================+=====================+
   | GNU 8.5.0 (serial mode)      |         ✓          |          ✓          |
   +------------------------------+--------------------+---------------------+
   | Intel 2020.2.0 (serial mode) |         ✓          |          ✘          |
   +------------------------------+--------------------+---------------------+
   | GNU 8.5.0 (OpenMP)           |         ✓          |          ✓          |
   +------------------------------+--------------------+---------------------+
   | Intel 2020.2.0 (OpenMP)      |         ✓          |          ✘          |
   +------------------------------+--------------------+---------------------+
