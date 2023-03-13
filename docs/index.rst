.. C++ Sphinx Doxygen Breathe documentation master file, created by
   sphinx-quickstart on Wed Jun 24 11:46:27 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPECFEM Kokkos documentation
=============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

About SPECFEM2D Kokkos
^^^^^^^^^^^^^^^^^^^^^^

Introduction
-------------

SPECFEM kokkos is C++ implementation of SPECFEM suite of software using the `Kokkos <https://kokkos.github.io/>`_ programming model. Kokkos is a is a production level solution for writing modern C++ applications in a hardware agnostic way, this allows us to write a single source code which can run across all modern architectures. The goal of this project is to provide the same level of functionality as provided by SPECFEM2D, SPECFEM3D and SPECFEM3d_GLOBE in a singular package that runs across all architectures.

Code feature matrix
--------------------

Table below shows various features available and tested in this package on various architectures:

+---------------------+-------------+-------------+------+-----+
|                     | CPU(serial) | CPU(OpenMP) | CUDA | HIP |
+=====================+=============+=============+======+=====+
| Physics                                                      |
+---------------------+-------------+-------------+------+-----+
| P-SV waves          | X           | X           | X    |     |
+---------------------+-------------+-------------+------+-----+
| Elastic Domains     | X           | X           | X    |     |
+---------------------+-------------+-------------+------+-----+
| Simulation Setup                                             |
+---------------------+-------------+-------------+------+-----+
| Forward Simulations | X           | X           | X    |     |
+---------------------+-------------+-------------+------+-----+
| Time Schemes                                                 |
+---------------------+-------------+-------------+------+-----+
| Newmark             | X           | X           | X    |     |
+---------------------+-------------+-------------+------+-----+
| Seismograms                                                  |
+---------------------+-------------+-------------+------+-----+
| displacement        | X           | X           | X    |     |
+---------------------+-------------+-------------+------+-----+
| velocity            | X           | X           | X    |     |
+---------------------+-------------+-------------+------+-----+
| Seimogram Formats                                            |
+---------------------+-------------+-------------+------+-----+
| ASCII               | X           | X           | X    |     |
+---------------------+-------------+-------------+------+-----+

.. note::
    While we work towards building this package and making the code/documentation more complete, please refer relevant SPECFEM package documentations for technical details on SPECFEM theory.

Any contributions to this documentation and package are always welcome. Please see :ref:`developer documentation` sections for more details.



Table of Contents
^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    user_documentation/index
    parameter_documentation/index
    source_description/index
    cookbooks/index
    developer_documentation/index
    api/index

Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
