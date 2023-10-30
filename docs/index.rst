SPECFEM++ - A modular and portable spectral-element code for seismic wave propagation
=====================================================================================

.. admonition:: Community Project

    SPECFEM++ is a community project. We welcome contributions from everyone. Please see :ref:`developer documentation` sections for more details.

.. admonition:: Under Development

    The package is currently under development and is not yet ready for production use. We are working towards making the package more complete. Please `report any bugs <https://github.com/PrincetonUniversity/specfem2d_kokkos/issues/new>`_ you find or `request features <https://github.com/PrincetonUniversity/specfem2d_kokkos/issues/new>`_ that you would like to see in the package.

SPECFEM++ is a complete re-write of SPECFEM suite of packages (SPECFEM2D, SPECFEM3D, SPECFEM3D_GLOBE) using C++. Compared to the earlier version, SPECFEM++ code base provides:

 1. a robust and flexible code structure,
 2. modularity that allows for easy addition of new features,
 3. portability that allows the code to run on a variety of architectures (CPU, NVIDIA GPUs, Intel GPUs, AMD GPUs etc.), and
 4. a user-friendly build infrastructure that allows the code to be easily compiled and run on a variety of platforms.

Current capabilities
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
 | Acoustic Domains    | ✔           | ✔           | ✔    |     |
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
    While we work towards building this package and making the code/documentation more complete, please refer relevant SPECFEM package documentations for technical details on SPECFEM theory.

Any contributions to this documentation and package are always welcome. Please see :ref:`developer documentation` sections for more details.

.. raw:: html

   <style>
   /* front page: hide chapter titles
    * needed for consistent HTML-PDF-EPUB chapters
    */
   section#user-documentation,
   section#cookbooks,
   section#contribution-guidelines,
   section#api-documentation,
   section#community {
       display:none;
   }
   </style>

User Documentation
------------------

.. toctree::
    :caption: USER DOCUMENTATION
    :maxdepth: 1
    :hidden:

    user_documentation/index
    meshfem2d/index
    parameter_documentation/index
    source_description/index

Cookbooks
---------

.. toctree::
    :caption: COOKBOOKS
    :maxdepth: 1
    :hidden:

    cookbooks/index

Contribution Guidelines
-----------------------

.. toctree::
    :caption: CONTRIBUTION GUIDELINES
    :maxdepth: 1
    :hidden:

    developer_documentation/index

Community
---------

.. toctree::
    :caption: COMMUNITY
    :maxdepth: 1
    :hidden:

    report_bugs/index
    feature_request/index
    join_the_community/index

API Documentation
-----------------

.. toctree::
    :caption: API DOCUMENTATION
    :maxdepth: 1
    :hidden:

    api/index
