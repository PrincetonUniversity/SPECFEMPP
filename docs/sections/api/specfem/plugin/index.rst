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
``specfem/plugin/injection_abi.h``. All C names use the ``specfempp_`` /
``SPECFEMPP_`` prefix, matching the exported entry point
``specfempp_injection_plugin``.

Points are delivered in groups, one per material system (medium and property
pair) of SPECFEM++, so a plugin sees the same medium x property structure as
the solver. Each group carries its material properties under the names used by
the SPECFEM++ point property containers (for example ``kappa``, ``mu`` and
``rho`` for elastic/isotropic, or ``c11`` ... ``c66`` and ``rho`` for
elastic/anisotropic), and the buffer layout of every requested quantity. All
values are in the units of the SPECFEM++ mesh and model; no unit conversion is
performed. ABI version 1 is three-dimensional only.

Every struct passed by pointer starts with ``struct_size``, so fields can be
appended without a new ABI version; use ``SPECFEMPP_HAS_FIELD`` before reading
a field added after version 1.

.. doxygenstruct:: specfempp_injection_plugin_table
    :members:

.. doxygenstruct:: specfempp_simulation_info
    :members:

.. doxygenstruct:: specfempp_communicator
    :members:

.. doxygenstruct:: specfempp_points
    :members:

.. doxygenstruct:: specfempp_point_group
    :members:

.. doxygenstruct:: specfempp_quantity_slot
    :members:

.. doxygenstruct:: specfempp_buffer
    :members:

.. doxygenstruct:: specfempp_manifest
    :members:

C++ layer
---------

.. doxygenclass:: specfem::plugin::InjectionPlugin
    :members:

.. doxygenclass:: specfem::plugin::Points
    :members:

.. doxygenclass:: specfem::plugin::PointGroup
    :members:

.. doxygenstruct:: specfem::plugin::QuantitySlot
    :members:

.. doxygenclass:: specfem::plugin::Buffer
    :members:

.. doxygenclass:: specfem::plugin::SimulationInfo
    :members:

.. doxygenclass:: specfem::plugin::Communicator
    :members:

.. doxygenclass:: specfem::plugin::Unsupported

.. doxygendefine:: SPECFEMPP_DEFINE_INJECTION_PLUGIN
