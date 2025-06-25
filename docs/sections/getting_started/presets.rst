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

For example, to configure and build the default release configuration:

.. code-block:: bash

    cmake --preset release
    cmake --build --preset release

To build with CUDA support:

.. code-block:: bash

    cmake --preset release-cuda
    cmake --build --preset release-cuda

The binaries built with and without CUDA will be generated in the ``bin/release`` and ``bin/release-cuda`` directories, respectively.
When running SPECFEM++, make sure you are using the correct binary for your chosen preset. You can either export the appropriate directory to your ``PATH`` environment variable, or run the executable by specifying its full path (e.g., ``<SPECFEMPP_DIR>/bin/release/specfem2d`` or ``<SPECFEMPP_DIR>/bin/release-cuda/specfem2d``).

Customizing Presets
-------------------

.. warning::

  Do not modify the provided ``CMakePresets.json`` file directly. Instead, create a ``CMakeUserPresets.json`` file in the root directory to add or override presets with your own custom configurations. This approach keeps your changes separate and avoids conflicts when updating the repository.

**Example: Creating a custom user preset**

1. Copy the structure below into a new file named ``CMakeUserPresets.json`` in the SPECFEM++ root directory:

.. code-block:: json

    {
      "version": 6,
      "configurePresets": [
        {
          "name": "my-custom-release",
          "inherits": "release",
          "cacheVariables": {
            "MY_OPTION": "ON"
          }
        }
      ],
      "buildPresets":[ {
          "name": "my-custom-release",
          "configurePreset": "my-custom-release",
          "targets": [
            "all"
          ]
        }
      ]
    }

2. Use your custom preset as usual:

.. code-block:: bash

    cmake --preset my-custom-release
    cmake --build --preset my-custom-release

For more details on CMake presets and user presets, see the `CMake documentation <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_.

**Summary:**
 * Use CMake presets to easily switch between build configurations.
 * Choose the correct path for your builds (e.g., ``bin/release`` for CPU, ``bin/release-cuda`` for CUDA).
 * Always create or modify ``CMakeUserPresets.json`` for your custom settings.
