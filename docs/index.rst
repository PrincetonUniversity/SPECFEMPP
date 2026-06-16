
.. meta::
  :hide-toc:

SPECFEM++ - A modular and portable spectral-element code for seismic wave propagation
=====================================================================================

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://github.com/PrincetonUniversity/SPECFEMPP/blob/main/LICENSE
    :alt: License

.. admonition:: Community Project

    SPECFEM++ is a community project. We welcome contributions from everyone. Please see contribution section for more details.

.. admonition:: Under Development

    The package is currently under development and is not yet ready for production use. We are working towards making the package more complete. Please `report any bugs <https://github.com/PrincetonUniversity/specfem2d_kokkos/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=>`_ you find or `request features <https://github.com/PrincetonUniversity/specfem2d_kokkos/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=>`_ that you would like to see in the package.

.. admonition:: Monthly Developer Meetings

    We host a monthly developer meeting on the first Wednesday of every month at 12:00 PM Eastern Time. Please join us if you are interested in contributing to the project or would like to learn more about the project. Please see `this page <https://github.com/orgs/SPECFEM/discussions>`_ for more details.

SPECFEM++ is a complete re-write of SPECFEM suite of packages (SPECFEM2D, SPECFEM3D, SPECFEM3D_GLOBE) using C++. Compared to the earlier version, SPECFEM++ code base provides:

 1. a robust and flexible code structure,
 2. modularity that allows for easy addition of new features,
 3. portability that allows the code to run on a variety of architectures (CPU, NVIDIA GPUs, Intel GPUs, AMD GPUs etc.), and
 4. a user-friendly build infrastructure that allows the code to be easily compiled and run on a variety of platforms.

.. admonition:: Download the code

    The code is available on `GitHub <https://github.com/PrincetonUniversity/specfem2d_kokkos/tree/main>`_.

Current capabilities
--------------------

Table below shows various features available and tested in this package on various architectures:


.. list-table:: Feature Support Matrix
    :widths: 36 16 16 16 16
    :header-rows: 1

    * - **Feature**
      - **CPU (serial)**
      - **CPU (OpenMP)**
      - **CUDA**
      - **HIP**

    * - **2-D Physics (Forward / Misfit Kernels)**
      -
      -
      -
      -

    * - Acoustic Isotropic
      - ✔ / ✔
      - ✔ / ✔
      - ✔ / ✔
      - ✔* / ✔*

    * - Elastic Isotropic P-SV
      - ✔ / ✔
      - ✔ / ✔
      - ✔ / ✔
      - ✔* / ✔*

    * - Elastic Isotropic SH
      - ✔ / ✔*
      - ✔ / ✔*
      - ✔ / ✔*
      - ✔* / ✔*

    * - Elastic Anisotropic P-SV
      - ✔ / ✔
      - ✔ / ✔
      - ✔ / ✔
      - ✔* / ✔*

    * - Elastic Anisotropic SH
      - ✔ / ✘
      - ✔ / ✘
      - ✔ / ✘
      - ✔* / ✘

    * - Poroelastic Isotropic (P-SV only)
      - ✔ /  ✘
      - ✔ /  ✘
      - ✔ /  ✘
      - ✔* / ✘

    * - Elastic Isotropic Cosserat P-SV **
      - ✔ /  ✘
      - ✔ /  ✘
      - ✔ /  ✘
      - ✔* / ✘

    * - **2-D Medium Coupling**
      -
      -
      -
      -

    * - Acoustic-Elastic coupling
      - ✔
      - ✔
      - ✔
      - ✔*

    * - Acoustic-Poroelastic coupling
      - ✘
      - ✘
      - ✘
      - ✘

    * - Elastic-Poroelastic coupling
      - ✘
      - ✘
      - ✘
      - ✘

    * - **Boundary Conditions (BC)**
      -
      -
      -
      -

    * - Absorbing BC (Stacey)
      - ✔
      - ✔
      - ✔
      - ✔*

    * - Free Surface BC
      - ✔
      - ✔
      - ✔
      - ✔*

    * - **Simulation Setup**
      -
      -
      -
      -

    * - Forward Simulations
      - ✔
      - ✔
      - ✔
      - ✔*

    * - Adjoint Simulations
      - ✔
      - ✔
      - ✔
      - ✔*

    * - **Time Schemes**
      -
      -
      -
      -

    * - Newmark
      - ✔
      - ✔
      - ✔
      - ✔*

    * - **Seismograms**
      -
      -
      -
      -

    * - Displacement
      - ✔
      - ✔
      - ✔
      - ✔*

    * - Velocity
      - ✔
      - ✔
      - ✔
      - ✔*

    * - Acceleration
      - ✔
      - ✔
      - ✔
      - ✔*

    * - **Seismogram Formats**
      -
      -
      -
      -

    * - ASCII
      - ✔
      - ✔
      - ✔
      - ✔*

\* Not tested. This, in general means that the feature is not tested in
continuous integration if the entire column has a star, and if the row has a
star, it means that the feature is not tested/does not match the `Fortran`
version of `specfem2d`. In the former case, see `HIP` column, and for the latter
case, see the Poroelastic Isotropic row.

\*\* Not peer reviewed yet.

.. note::

    While we work towards building this package and making the
    code/documentation more complete, please refer relevant SPECFEM package
    documentations for technical details on SPECFEM theory.

.. raw:: html

   <style>
   /* front page: hide chapter titles
    * needed for consistent HTML-PDF-EPUB chapters
    */
   section#getting-started,
   section#user-documentation,
   section#cookbooks,
   section#contribution,
   section#api-documentation,
   section#benchmarks,
   section#community {
       display:none;
   }
   </style>

Getting Started
---------------

.. toctree::
    :caption: GETTING STARTED
    :maxdepth: 1
    :hidden:

    sections/getting_started/index
    sections/cookbooks/index
    sections/getting_started/presets
    sections/getting_started/special_machines

User Documentation
------------------

.. toctree::
    :caption: USER DOCUMENTATION
    :maxdepth: 2
    :hidden:

    sections/meshfem/index
    sections/parameter_documentation/index
    sections/source_description/index

Contribution
------------

.. toctree::
    :caption: CONTRIBUTION
    :maxdepth: 1
    :hidden:

    sections/developer_documentation/contributing
    sections/developer_documentation/style
    sections/developer_documentation/git_workflow
    sections/developer_documentation/build_requirements
    sections/developer_documentation/continuous_integration

.. sections/developer_documentation/tutorials/index

Community
---------

.. toctree::
    :caption: COMMUNITY
    :maxdepth: 1
    :hidden:

    Report bugs <https://github.com/PrincetonUniversity/specfempp/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=>
    Request a new feature <https://github.com/PrincetonUniversity/specfempp/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=>
    Join the discussion <https://github.com/orgs/SPECFEM/discussions>

Benchmarks
----------

.. toctree::
    :caption: BENCHMARKS
    :maxdepth: 1
    :hidden:

    Forward simulations <https://github.com/PrincetonUniversity/SPECFEMPP-benchmarks/blob/main/forward_simulations/README.md>

API Documentation
-----------------

.. toctree::
    :caption: API DOCUMENTATION
    :maxdepth: 1
    :hidden:

    sections/api/index
