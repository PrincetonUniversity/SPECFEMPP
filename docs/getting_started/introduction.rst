.. This file is not part of the SPECFEM++ documentation.

Introduction
============

SPECFEM kokkos is C++ implementation of SPECFEM suite of software using the `Kokkos <https://kokkos.github.io/>`_ programming model. Kokkos is a is a production level solution for writing modern C++ applications in a hardware agnostic way, this allows us to write a single source code which can run across all modern architectures. The goal of this project is to provide the same level of functionality as provided by SPECFEM2D, SPECFEM3D and SPECFEM3GLOBE in a singular package that runs across all architectures.

Code feature matrix
--------------------

Table below shows various features available and tested in this package on various architectures:

+---------------------+-------------+-------------+------+-----+
|                     | CPU(serial) | CPU(OpenMP) | CUDA | HIP |
+=====================+=============+=============+======+=====+
| Physics                                                      |
+---------------------+-------------+-------------+------+-----+
| P-SV waves          | ✔           | ✔           | ✔    |     |
+---------------------+-------------+-------------+------+-----+
| Elastic Domains     | ✔           | ✔           | ✔    |     |
+---------------------+-------------+-------------+------+-----+
| Simulation Setup                                             |
+---------------------+-------------+-------------+------+-----+
| Forward Simulations | ✔           | ✔           | ✔    |     |
+---------------------+-------------+-------------+------+-----+
| Time Schemes                                                 |
+---------------------+-------------+-------------+------+-----+
| Newmark             | ✔           | ✔           | ✔    |     |
+---------------------+-------------+-------------+------+-----+
| Seismograms                                                  |
+---------------------+-------------+-------------+------+-----+
| displacement        | ✔           | ✔           | ✔    |     |
+---------------------+-------------+-------------+------+-----+
| velocity            | ✔           | ✔           | ✔    |     |
+---------------------+-------------+-------------+------+-----+
| Seimogram Formats                                            |
+---------------------+-------------+-------------+------+-----+
| ASCII               | ✔           | ✔           | ✔    |     |
+---------------------+-------------+-------------+------+-----+

.. note::

    While we work towards building this package and making the code/documentation more complete, please refer relevant
    SPECFEM package documentations for technical details on SPECFEM theory.

Any contributions to this documentation and package are always welcome. Please see <contributions> sections for more details.
