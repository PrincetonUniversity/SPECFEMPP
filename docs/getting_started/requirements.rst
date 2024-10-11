Requirements
=============

Compiler Versions
-----------------

.. note::
    The following compilers are supported and tested by Kokkos. In theory, SPECFEM++ should work with any of these compiler versions. However we have not tested all of them and cannot guarantee the same. If you have issues compiling with a compiler versions listed below, please create an issue on GitHub. For a list of tested compilers and platforms, see :ref:`tests` section.

.. list-table::
    :widths: 30 35 35
    :header-rows: 1
    :align: center

    * - Compiler
      - Minimum version
      - Primary tested versions

    * * GCC
      * 5.3.0
      * 5.3.0, 6.1.0, 7.3.0, 8.3, 9.2, 10.0

    * * Clang
      * 4.0.0
      * 8.0.0, 9.0.0, 10.0.0, 12.0.0

    * * Intel
      * 17.0.1
      * 17.4, 18.1, 19.5

    * * NVCC
      * 9.2.88
      * 9.2.88, 10.1, 11.0

    * * NVC++
      * 21.5
      * NA

    * * ROCM
      * 4.5
      * 4.5.0

    * * ARM/Clang
      * 20.1
      * 20.1

.. warning::

    We support cudatoolkit versions ``>=11.7``. For a list of tested compilers and platforms, see :ref:`tests` section.

Build system
------------

* CMake >= 3.16: required
* CMake >= 3.21.1 for NVC++


Dependencies
------------

None of the dependencies need to be installed prior to the installation of
the package. Having installed some packages does however reduce build time
because the dependencies do not have to be fetched.

Boost
+++++

Current requirements of the ``Boost`` library are a version that is ``>=1.66.0``.
If you have ``Boost`` installed on your system, set the environment variable
``BOOST_ROOT`` containing the absolute path to your ``Boost`` installation.
For example, on machines (clusters) with ``lmod`` package manager this can be
done by loading the boost module

.. code:: bash

    module load boost/?.??.? # Eg. boost/1.73.0
