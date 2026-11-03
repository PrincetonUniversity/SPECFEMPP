Build system requirements
=========================

This section covers compilation checks required so that SPECFEM++ is able to run
across all architectures.

.. note::

    This section is still under development.

Optional dependencies
----------------------

* **Trilinos** -- enables implicit (matrix-assembled) solvers. See
  :ref:`trilinos_configuration` for how to build SPECFEM++ against a shared
  Trilinos install. Trilinos must share the same Kokkos as SPECFEM++.

Intel GPU ahead-of-time compilation
----------------------------------

The ``release-sycl`` preset uses portable SPIR-V with just-in-time compilation.
To compile native GPU code for your local Intel GPU, load your Intel oneAPI
compiler environment, install the Intel offline compiler (``ocloc``), and run
from the repository root:

.. code-block:: bash

   python3 scripts/setup_sycl_aot.py --configure
   cmake --build --preset release-sycl-aot --parallel 4

The script detects the GPU's PCI device ID through Linux sysfs and compiles a
small SYCL kernel to check that the installed compiler supports the target.
It does not execute the kernel. After successful validation, it creates or
updates ``release-sycl-aot`` in the gitignored ``CMakeUserPresets.json``,
preserving unrelated presets and settings. The AOT build uses its own build,
installation, and benchmark directories. Omit ``--configure`` to generate the
presets without configuring the project. Use ``--compiler /path/to/icpx`` if
the compiler is not on your executable search path.

If several distinct Intel GPU device IDs are detected, the script lists them
and requires an explicit selection. For cross-compilation, or on systems without
Linux sysfs, supply the destination GPU's PCI device ID:

.. code-block:: bash

   python3 scripts/setup_sycl_aot.py --device 0x7d51 --configure

Replace ``0x7d51`` with the destination device ID. No GPU architecture name is
needed. Native binaries target the selected GPU; use ``release-sycl`` for portable
JIT compilation instead. AOT moves device compilation to build time and requires
a compatible driver at runtime. Changing the selected target requires rebuilding.
