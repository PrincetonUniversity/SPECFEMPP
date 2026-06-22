CMake Presets for Multiple Build Configurations
===============================================

When working with SPECFEM++, you may need to switch between different build configurations, such as CPU and GPU builds. CMake presets provide a convenient way to manage these configurations without manually specifying all the build options each time.

Using CMake Presets
-------------------

SPECFEM++ provides a set of default presets in the ``CMakePresets.json`` file located in the root directory of the repository. These presets cover common configurations, including CPU (with or without SIMD), CUDA, and HIP builds.

**To use a preset, run:**

.. code-block:: bash

    cmake --preset <preset-name>
    cmake --build --preset <preset-name>

.. note::

    To list all available presets, you can run :code:`cmake --list-presets`.

For example, to configure and build the default release configuration:

.. code-block:: bash

    cmake --preset release
    cmake --build --preset release

To build with CUDA support:

.. code-block:: bash

    cmake --preset release-cuda
    cmake --build --preset release-cuda

The binaries built with and without CUDA will be generated in the ``bin/release`` and ``bin/release-cuda`` directories, respectively.
When running SPECFEM++, make sure you are using the correct binary for your chosen preset. You can either export the appropriate directory to your ``PATH`` environment variable, or run the executable by specifying its full path (e.g., ``<SPECFEMPP_DIR>/bin/release/specfem`` or ``<SPECFEMPP_DIR>/bin/release-cuda/specfem``).

.. note::

    SPECFEM++ presets use CMake's Unity build feature to speed up compilation times. However, this can lead to increased memory usage during the build process. In some cases, especially on systems with limited RAM, you may encounter out-of-memory errors during compilation. If this happens, you can reduce the unity build batch size by passing ``-D SPECFEM_UNITY_BUILD_BATCH_SIZE=<smaller_number>`` (for example, ``-D SPECFEM_UNITY_BUILD_BATCH_SIZE=4``) when invoking CMake with the preset. This will reduce memory consumption.

Customizing Presets
-------------------

If the provided presets do not meet your specific needs, you can create your own custom presets by defining them in a ``CMakeUserPresets.json`` file in the root directory of the SPECFEM++ repository. This file allows you to override or extend the default presets without modifying the original ``CMakePresets.json``.

Below is an example of how to create a custom preset that compiles SPECFEM++ for NVIDIA A100 GPUs with an additional user-defined option:

.. code-block:: json
    :caption: CMakeUserPresets.json

    {
      "version": 6,
      "configurePresets": [
        {
          "name": "release-ampere80",
          "inherits": "release",
          "cacheVariables": {
            "Kokkos_ARCH_AMPERE80": "ON",
            "Kokkos_ARCH_NATIVE": "OFF",
          }
        }
      ],
      "buildPresets":[ {
          "name": "release-ampere80",
          "configurePreset": "release-ampere80",
          "targets": [
            "all"
          ]
        }
      ]
    }

For more details on CMake presets and user presets, see the `CMake documentation <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_.

**Summary:**
 * Use CMake presets to easily switch between build configurations.
 * Choose the correct path for your builds (e.g., ``bin/release`` for CPU, ``bin/release-cuda`` for CUDA).
 * Always create or modify ``CMakeUserPresets.json`` for your custom settings.
