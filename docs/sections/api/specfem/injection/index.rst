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

Provider session
----------------

The provider session is the SPECFEM++ side of the plugin ABI: it drives a
loaded plugin through its lifecycle and serves the incident field on the
device, one aligned window of time steps at a time. It uses Kokkos and MPI;
plugins never see it, and receive only plain C arrays and a Fortran
communicator handle.

Injection points are ordered by point group, one group per material system
(medium and property, e.g. elastic/isotropic). Each group carries its
material properties by name and the quantities requested there, with the
number of buffer components each occupies. Every rank holds the same groups in
the same order, including groups with no points on that rank. The plugin fills
one buffer per group, each a strided view of the group's rows in the frame
buffer, so ``FrameBuffer::load(istep, row, slot)`` reads any point the same
way.

.. doxygenclass:: specfem::injection::ProviderSession
    :members:

.. doxygenstruct:: specfem::injection::SimulationParameters
    :members:

.. doxygenstruct:: specfem::injection::SessionOptions
    :members:

.. doxygenstruct:: specfem::injection::InjectionPoints
    :members:

.. doxygenstruct:: specfem::injection::PointGroup
    :members:

.. doxygenstruct:: specfem::injection::QuantityRequest
    :members:

.. doxygenstruct:: specfem::injection::FrameBuffer
    :members:

.. doxygenfunction:: specfem::injection::window_start

.. doxygenfunction:: specfem::injection::window_steps_for_budget

Injection points
----------------

:cpp:func:`specfem::injection::make_injection_points` turns the assembled
injection surface (:doc:`../assembly/injection_surface/injection_surface`) into
the points for the provider session. Every material system of the solver
becomes a point group, in the same order on every rank. The injection method
decides, through a :cpp:type:`specfem::injection::QuantityPolicy`, which
quantities each system requests; it is only asked about systems that have
injection points on some rank, and a system it does not support fails the run
on every rank. Coordinates and normals are shared with the assembly rather than
copied, and material properties are passed by name.

.. doxygentypedef:: specfem::injection::QuantityPolicy

.. doxygenfunction:: specfem::injection::make_injection_points

.. doxygenfunction:: specfem::injection::make_simulation_parameters

Methods
-------

An injection method defines how the solver applies the incident field. The
time-marching solver sees only :cpp:class:`specfem::injection::CouplingMethod`:
each step it calls ``prepare_step(istep)`` once, then ``apply`` for every
medium right after the source interaction, before the stiffness kernels, the
MPI exchange and the mass-matrix division. Step ``istep`` holds the incident
field at simulation time :math:`t_0 + \text{istep}\,\Delta t`.

The program builds the method named in ``parameters.injection.method`` from a
:cpp:class:`specfem::injection::MethodRegistry`, starts the plugin's provider
session, sets it up with the method's injection points, and hands the session
to the method (:cpp:class:`specfem::injection::InjectionMethod`). Methods built
into SPECFEM++ are listed in ``default_method_registry()``; tests register their
own.

.. doxygenclass:: specfem::injection::CouplingMethod
    :members:

.. doxygenclass:: specfem::injection::InjectionMethod
    :members:

.. doxygenclass:: specfem::injection::MethodRegistry
    :members:

.. doxygenfunction:: specfem::injection::default_method_registry

Pointwise surface methods
-------------------------

Most methods add a term computed independently at every injection point.
:cpp:class:`specfem::injection::PointwiseSurfaceMethod` implements the whole
method interface for them: the author writes a struct with a name, the
quantities requested per medium, and a static ``compute`` per supported
material system, and registers it wrapped in ``PointwiseSurfaceMethod``.
Deriving the struct from :cpp:struct:`specfem::injection::PointwiseMethodBase`
provides the shorthand ``Context<Medium, Property>`` for its point context.

.. code-block:: cpp

    struct ConstantForce
        : specfem::injection::PointwiseMethodBase<ConstantForce> {
      static constexpr std::string_view method_name = "constant-force";

      struct Parameters {
        type_real scale = 1.0;
      };

      using elastic_quantities =
          specfem::injection::QuantityList<specfem_quantity_velocity>;

      KOKKOS_INLINE_FUNCTION static specfem::point::acceleration<
          specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                              specfem::element::medium_tag::elastic,
                              specfem::element::property_tag::isotropic, false>>
      compute(const Context<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic> &ctx) {
        const auto v = ctx.incident<specfem_quantity_velocity>();
        const type_real factor = ctx.parameters().scale * ctx.weight();
        return { factor * v(0), factor * v(1), factor * v(2) };
      }
    };

    registry.add<specfem::injection::PointwiseSurfaceMethod<ConstantForce>>();

The material systems a method supports are those it has a ``compute`` for; the
plugin is asked for the quantities of those systems only, with the number of
components each occupies taken from the same table that places them in the
frame buffer. ``Parameters``, if present, are default-constructed.

``PointwiseSurfaceMethod`` constrains its template parameter with the concept
``specfem::injection_impl::PointwiseInjectionMethod``; the compiler names the
first requirement the method does not satisfy. A method must:

- declare ``static constexpr std::string_view method_name``;
- have at least one static ``compute`` taking a ``const PointContext<Method,
  Medium, Property> &``, each returning the point acceleration of its material system;
- declare a quantity list for a medium exactly when it has a ``compute`` for
  that medium;
- list each quantity at most once, and only quantities its medium provides;
- make ``Parameters``, if present, trivially copyable and
  default-constructible.

Each step, ``apply`` runs one kernel per supported material system over its
entries of the injection surface. For each entry it loads the solver's
material properties at the point, builds a
:cpp:class:`specfem::injection::PointContext`, calls ``compute``, and adds the
result to the forward acceleration atomically, since a GLL point can be shared
by several elements and faces. Points where the acceleration is held at zero
(an acoustic free surface) receive nothing. Only the forward wavefield is
supported.

:cpp:func:`specfem::injection::evaluate_point` runs ``compute`` at a single
point described by explicit values, so a method's physics can be unit-tested
without a mesh.

.. doxygenclass:: specfem::injection::PointwiseSurfaceMethod
    :members:

.. doxygenclass:: specfem::injection::PointContext
    :members:

.. doxygenstruct:: specfem::injection::PointwiseMethodBase

.. doxygenstruct:: specfem::injection::QuantityList

.. doxygenfunction:: specfem::injection::quantity_components

.. doxygenstruct:: specfem::injection::QuantitySlot

.. doxygenstruct:: specfem::injection::QuantityListTraits

.. doxygenfunction:: specfem::injection::to_requests

.. doxygenfunction:: specfem::injection::evaluate_point

Stacey injection
~~~~~~~~~~~~~~~~

``method: stacey`` selects :cpp:struct:`specfem::injection::StaceyInjection`,
for elastic isotropic and anisotropic points. At a Stacey point the stiffness
kernel adds :math:`-Z(v)\,dS`; the method adds

.. math::

    f = \left(t^* + Z(v^*)\right) dS, \qquad
    Z(v) = (\rho v_p - \rho v_s)(v \cdot \hat{n})\,\hat{n} + \rho v_s\, v,

so the boundary absorbs only the field scattered away from the incident one.
It requests the incident velocity and traction from the plugin. Anisotropic
points use the Voigt-averaged impedances, as Stacey's absorbing term does.
The injection points must lie on Stacey boundaries; the method does not check
this.

.. doxygenstruct:: specfem::injection::StaceyInjection
