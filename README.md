# SPECFEM++

[![Documentation Status](https://readthedocs.org/projects/specfem2d-kokkos/badge/?version=latest)](https://specfem2d-kokkos.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![NVIDIA Build](https://jenkins.princeton.edu/buildStatus/icon?job=SpecFEM_KOKKOS%2FNVIDIA_Compiler_Checks&build=last&subject=NVIDIA%20Build)
![GCC Build](https://jenkins.princeton.edu/buildStatus/icon?job=SpecFEM_KOKKOS%2FGNU+Compiler&build=last&subject=GCC%20Build)
![IntelLLVM Build](https://jenkins.princeton.edu/buildStatus/icon?job=SpecFEM_KOKKOS%2FIntel_Compiler_Checks&build=last&subject=IntelLLVM%20Build)


## About

SPECFEM++ is a complete re-write of SPECFEM suite of packages (SPECFEM2D, SPECFEM3D, SPECFEM3D_GLOBE) using C++. Compared to the earlier version, SPECFEM++ code base provides:

 1. a robust and flexible code structure,
 2. modularity that allows for easy addition of new features,
 3. portability that allows the code to run on a variety of architectures (CPU, NVIDIA GPUs, Intel GPUs, AMD GPUs etc.), and
 4. a user-friendly build infrastructure that allows the code to be easily compiled and run on a variety of platforms.

[specfempp-py](https://github.com/PrincetonUniversity/SPECFEMPP-py) is the official Python package for configuring and running SPECFEM++ with Python.

## Documentation


The online documentation for SPECFEM++ is located [here](https://specfem2d-kokkos.readthedocs.io/en/latest/index.html#)

## Getting Started with SPECFEM++

Follow the [Getting Started Guide](https://specfem2d-kokkos.readthedocs.io/en/latest/getting_started/index.html) to install SPECFEM++ on your system and run the solver.

## Examples

We recommend starting with the [cookbook examples](https://specfem2d-kokkos.readthedocs.io/en/latest/cookbooks/index.html) to learn how to customize the solver for your use case.

## Contributing to SPECFEM++

SPECFEM is a community project that lives by the participation of its members — i.e., including you! It is our goal to build an inclusive and participatory community so we are happy that you are interested in participating! Please see [this page](https://specfem2d-kokkos.readthedocs.io/en/latest/developer_documentation/index.html) for developer documentation.

In particular you should follow the git development workflow and pre-commit style checks when contributing to SPECEFM.

## License

SPECFEM++ is distributed under the [GPL v3 license](LICENSE)
