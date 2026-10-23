``specfem::plugin``
===================

Header-only SDK for wavefield-injection plugins: shared libraries that provide
an incident wavefield to SPECFEM++ at runtime. The SDK lives in ``sdk/`` and is
independent of the SPECFEM++ core (no Kokkos, MPI, or yaml-cpp). Install it
with ``cmake --install <build> --component plugin_sdk --prefix <dir>`` and use
it from a plugin project with ``find_package(SpecfemppPluginSDK)`` and
``specfempp_add_injection_plugin(<target> SOURCES ...)``.

.. doxygennamespace:: specfem::plugin
    :desc-only:

C ABI
-----

Everything that crosses the library boundary is defined in
``specfem/plugin/injection_abi.h``.

.. doxygenstruct:: spfp_injection_plugin
    :members:

.. doxygenstruct:: spfp_simulation_info
    :members:

.. doxygenstruct:: spfp_communicator
    :members:

.. doxygenstruct:: spfp_points
    :members:

.. doxygenstruct:: spfp_request
    :members:

.. doxygenstruct:: spfp_buffer
    :members:

.. doxygenstruct:: spfp_manifest
    :members:

C++ layer
---------

.. doxygenclass:: specfem::plugin::InjectionPlugin
    :members:

.. doxygenclass:: specfem::plugin::Points
    :members:

.. doxygenclass:: specfem::plugin::Request
    :members:

.. doxygenclass:: specfem::plugin::Buffer
    :members:

.. doxygenclass:: specfem::plugin::SimulationInfo
    :members:

.. doxygenclass:: specfem::plugin::Communicator
    :members:

.. doxygendefine:: SPECFEMPP_DEFINE_INJECTION_PLUGIN
