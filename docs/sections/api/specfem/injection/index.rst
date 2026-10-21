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
