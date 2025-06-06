CMake
=====

.. note::

    This section is still under development. For now, please refer to the
    `CMake documentation <https://cmake.org/cmake/help/latest/index.html>`_
    for more information.


We are using the `CMake <https://cmake.org/>`_ build system to manage the
build process. CMake is a cross-platform, open-source build system that
uses a simple configuration file, ``CMakeLists.txt``, to generate
build files for different platforms and compilers. CMake is widely used
in the scientific computing community and is a standard for many
scientific software packages. It is a powerful tool that allows for
flexible and efficient builds, making it an ideal choice for
SPECFEM++.

For the user, CMake provides a simple and consistent interface to
configure and build the software. It allows users to specify
dependencies, compiler options, and build targets in a single
configuration file. CMake then generates the appropriate build files
for the user's platform and compiler, making it easy to build the
software on different systems.

Simple default build
--------------------

The simplest way to build SPECFEM++ is to run the following command

.. code-block:: bash

    # Configuration
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release

    # Build
    cmake --build build

This will create a build directory called ``build`` and generate the build files
for the specified build type. All the CMake relevant files, such as build cash
are compiled binary files, will be stored in the ``build`` directory.

At the end of the configuration step, CMake will print out the configuration
summary, which includes the build type, compiler options, and any dependencies
that were found. This summary is useful for debugging and verifying that the
configuration was successful.

At the end of the build step, CMake will print out the build summary, which
includes the number of files that were built. It will also print out that all
executables are put into the ``path/to/specfempp/bin`` directory. So that you can
run the executables from there. You can also add this directory to your ``PATH``

.. code-block:: bash

    export PATH=$(pwd)/bin:$PATH

and run the executables from anywhere.

The executables folder
----------------------



The path to to the installation directory can be set using the
``CMAKE_INSTALL_PREFIX`` variable. For example, to install SPECFEM++ in the
``/usr/local/bin`` directory, you can run the following command:

.. code-block:: bash

    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/bin

.. note::

    To install SPECFEM++ in a system directory, you may need to run the
    configuration step with `sudo` privileges.


Benchmarks folder
-----------------

As part of the configuration step, we build a directory of benchmarks. This
directory contains a set of benchmarks that can be used to check the solver
against reference solutions. The benchmarks are located in the ``benchmarks/src``
and are built by default under ``benchmarks/build``.

The path to the benchmarks build folder can be set using the
``BENCHMARKS_BUILD_DIR`` variable. For example, to install the benchmarks in the
``my/path/to/benchmarks`` directory, you can run the following command:

.. code-block:: bash

    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D BENCHMARKS_BUILD_DIR=my/path/to/benchmarks

This setup is particularly useful for testing results between different
version/architectures of SPECFEM++. The benchmarks folders are created to
be automatically run using
`snakemake <https://snakemake.readthedocs.io/en/stable/>`_.

So that in each benchmark folder (e.g.
``benchmarks/build/<myarch1>/dim2/benchmark1``), you can run the following
command to run the benchmark:

.. code-block:: bash

    snakemake -j 1

and run another benchmark (e.g. ``benchmarks/build/<myarch2>/dim2/benchmark1``)
using the same command, and compare the results, that is kernels/seismograms
etc.
