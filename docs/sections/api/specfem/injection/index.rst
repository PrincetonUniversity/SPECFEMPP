``specfem::injection``
======================

Loads wavefield-injection plugins built with the plugin SDK
(:doc:`../plugin/index`). A plugin is selected on the ``3d`` subcommand with
``--injection-plugin`` (an absolute path to a shared library, or
``static:<name>`` for a statically registered plugin; the
``SPECFEMPP_INJECTION_PLUGIN`` environment variable is the fallback) together
with ``--injection-config``, and requires a ``parameters.injection`` section in
the parameter file.

Before a shared library is loaded, its path must be absolute, the file and its
directory must be owned by the current user or root and not be group- or
world-writable, and the file must embed exactly one manifest for the supported
ABI version. Only then is it opened with ``dlopen``.

.. doxygennamespace:: specfem::injection
    :desc-only:

Plugins
-------

.. doxygenclass:: specfem::injection::PluginLibrary
    :members:

.. doxygenclass:: specfem::injection::PluginInstance
    :members:

.. doxygenfunction:: specfem::injection::register_static_plugin

Options
-------

.. doxygenstruct:: specfem::injection::PluginOptions
    :members:

.. doxygenfunction:: specfem::injection::resolve_plugin_options

.. doxygenfunction:: specfem::injection::check_injection_consistency

Request
-------

.. doxygenstruct:: specfem::injection::Request
    :members:

.. doxygenfunction:: specfem::injection::component_count

.. doxygenfunction:: specfem::injection::count_components

.. doxygenvariable:: specfem::injection::media

Provider session
----------------

The provider session is the SPECFEM++ side of the plugin ABI: it drives a
loaded plugin through its lifecycle and serves the incident field on the
device, one aligned window of time steps at a time. It uses Kokkos and MPI;
plugins never see it, and receive only plain C arrays and a Fortran
communicator handle.

.. doxygenclass:: specfem::injection::ProviderSession
    :members:

.. doxygenstruct:: specfem::injection::SimulationParameters
    :members:

.. doxygenstruct:: specfem::injection::SessionOptions
    :members:

.. doxygenstruct:: specfem::injection::InjectionPoints
    :members:

.. doxygenstruct:: specfem::injection::FrameBuffer
    :members:

.. doxygenfunction:: specfem::injection::window_start

.. doxygenfunction:: specfem::injection::window_steps_for_budget
