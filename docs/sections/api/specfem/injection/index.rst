``specfem::injection``
======================

Loads wavefield-injection plugins built with the plugin SDK
(:doc:`../plugin/index`). A plugin is selected on the ``3d`` subcommand with
``--injection-plugin`` (a path to a shared library, or
``static:<name>`` for a statically registered plugin) together with
``--injection-config``, and requires a ``parameters.injection`` section in
the parameter file.

Before a shared library is loaded, its path is resolved with ``realpath`` and
must name a regular file that embeds exactly one manifest for the supported ABI
version. Only the resolved absolute path is passed to ``dlopen``, so library
search paths are never used. The SHA-256 of the file is logged with the plugin's
name and version, to record which build ran.

Platforms
---------

Shared-library plugins are supported on Linux and macOS. On Windows, SPECFEM++
builds and runs, but ``--injection-plugin`` accepts only ``static:<name>``; a
path to a shared library is rejected with an error saying so, and
``specfem_add_injection_plugin()`` refuses to configure.

Security
--------

A plugin runs inside ``specfem`` with the rights of the process, so the
permission checks depend on whether that process is privileged.

**Normal runs.** The plugin runs as the user who started the run, who chose
the plugin, so it gains no rights the user did not already have. The only
outsiders worth keeping out are users who could not otherwise run code as this
user. The file must therefore not be world-writable, and its directory must not
be world-writable unless it has the sticky bit (as ``/tmp`` does). Plugins
owned by another user, or kept in a group-writable project directory, are
accepted: group members who can change the plugin can already change the run's
job scripts and inputs.

**Privileged runs** (effective user root, or a setuid ``specfem``). Anyone who
can change the plugin would gain those rights. The file and every directory
above it must be owned by root or the effective user, the file must not be
group- or world-writable, and no directory above it may be group- or
world-writable unless it has the sticky bit. This is the case for containers
that run as root: install plugins into a root-owned location in the image.

These checks guard against accidental exposure, not against an attacker who
already controls the user's account or environment (who could set
``LD_PRELOAD`` instead).

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
