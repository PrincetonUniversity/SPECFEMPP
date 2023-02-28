# SPECFEM2D Kokkos implementation

[![Unittests](https://github.com/PrincetonUniversity/specfem2d_kokkos/actions/workflows/unittests.yml/badge.svg)](https://github.com/PrincetonUniversity/specfem2d_kokkos/actions/workflows/unittests.yml)
[![Build](https://github.com/PrincetonUniversity/specfem2d_kokkos/actions/workflows/compilation.yml/badge.svg)](https://github.com/PrincetonUniversity/specfem2d_kokkos/actions/workflows/compilation.yml)
[![Documentation Status](https://readthedocs.org/projects/specfem2d-kokkos/badge/?version=latest)](https://specfem2d-kokkos.readthedocs.io/en/latest/?badge=latest)

## About


SPECFEM kokkos is C++ implementation of SPECFEM suite of software using the [Kokkos](<https://kokkos.github.io/>) programming model. Kokkos is a is a production level solution for writing modern C++ applications in a hardware agnostic way, this allows us to write a single source code which can run across all modern architectures. The goal of this project is to provide the same level of functionality as provided by SPECFEM2D, SPECFEM3D and SPECFEM3d_GLOBE in a singular package that runs across all architectures.

## Documentation


The online documentation for SPECFEM2D Kokkos is located [here](https://specfem2d-kokkos.readthedocs.io/en/latest/index.html#)

## Installation


Completer installation instructions are located in the [online documentation](https://specfem2d-kokkos.readthedocs.io/en/latest/user_documentation/index.html)

## Code feature matrix


Table below shows various features available and tested in this package on various architectures:

|                     | CPU(serial) | CPU(OpenMP) | CUDA | HIP
----------------------|------------:|-----------:|------:|-----|
| **Physics**                                                  |
| P-SV waves          | X           | X           | X    |     |
| SH waves            |             |             |      |     |
| Elastic Domains     | X           | X           | X    |     |
| **Simulation Setup**                                         |
| Forward Simulations | X           | X           | X    |     |
| **Time Schemes**                                             |
| Newmark             | X           | X           | X    |     |
| **Seismograms**                                              |
|                     |             |             |      |     |

## Running SPECFEM2D Kokkos

Intructions on how to run SPECFEM2D Kokkos can be found [here](https://specfem2d-kokkos.readthedocs.io/en/latest/user_documentation/index.html).

More use case examples of running the software please see [cookbooks](https://specfem2d-kokkos.readthedocs.io/en/latest/cookbooks/index.html)

## Contributing to SPECFEM2D Kokkos


SPECFEM is a community project that lives by the participation of its members — i.e., including you! It is our goal to build an inclusive and participatory community so we are happy that you are interested in participating! Please see [this page](https://specfem2d-kokkos.readthedocs.io/en/latest/developer_documentation/index.html) for developer documentation.

In particular you should follow the git development workflow and pre-commit style checks when contributing to SPECEFM.

## License


[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

SPECFEM2D Kokkos is distributed under the [GPL v3 license](LICENSE)
