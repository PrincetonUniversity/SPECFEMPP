``specfem::plugin``
===================

Header-only SDK for wavefield-injection plugins: shared libraries that provide
an incident wavefield to SPECFEM++ at runtime. The SDK lives in ``sdk/`` and is
independent of the SPECFEM++ core (no Kokkos, MPI, or yaml-cpp). Install it
with ``cmake --install <build> --component plugin_sdk --prefix <dir>`` and use
it from a plugin project with ``find_package(SpecfemPluginSDK)`` and
``specfem_add_injection_plugin(<target> SOURCES ...)``.

.. doxygennamespace:: specfem::plugin
    :desc-only:

Lifecycle and conventions
-------------------------

A plugin is a shared library that exports exactly one symbol,
``specfem_injection_plugin()``, returning a table of plain C function
pointers (:cpp:struct:`specfem_injection_plugin_table`). Everything that
crosses the library boundary is a C struct or a flat host array, so a plugin
can be built with any compiler and standard library, and may link its own
copies of third-party libraries. ``specfem_add_injection_plugin`` hides every
other symbol and checks the exported set after each build.

SPECFEM++ drives a plugin through the following calls::

    create -> supports (per medium, property and quantity) -> setup (once)
           -> fill (once per window of time steps) -> destroy

- **Collective calls.** Every call is made on all MPI ranks, including ranks
  that own no injection points. ``supports`` is queried for the global set of
  material systems, every rank receives the same list of point groups (groups
  without local points have ``count == 0``), and a single ``fill`` call fills
  every group. A plugin may therefore use collective MPI operations in any
  call.
- **Dimension.** ABI version 1 is three-dimensional only;
  ``specfem_simulation_info::dimension`` is 3.
- **Units.** Every value (coordinates, bounds, times, material properties,
  injected fields) is in the units of the SPECFEM++ mesh and model. SPECFEM++
  performs no unit conversion; configure the plugin to match.
- **Frame and time.** Coordinates are in the SPECFEM++ mesh frame (z up).
  Simulation step ``k`` is at time ``t0 + k * dt``. The bounds are those of the
  global domain, identical on every rank.
- **Lifetimes.** :cpp:struct:`specfem_points` and everything it points to is
  valid only during ``setup``; the buffers only during ``fill``. Copy what is
  needed.
- **Determinism.** ``fill`` must produce the same values regardless of window
  order or size.
- **Errors.** Errors are reported through status codes and ``last_error``.
  Exceptions must never cross the boundary, and plugins must not call
  ``exit``, ``abort``, ``MPI_Init`` or ``MPI_Finalize``. A plugin that cannot
  serve the run returns ``specfem_status_unsupported`` (in C++: throws
  :cpp:class:`specfem::plugin::Unsupported`); SPECFEM++ then stops with the
  plugin's message.
- **Manifest.** Every plugin embeds a :cpp:struct:`specfem_manifest` that
  SPECFEM++ reads from the file before loading it, so a plugin built for
  another ABI version never runs code in the simulation process.

Extending the ABI
~~~~~~~~~~~~~~~~~

Every struct passed by pointer starts with ``struct_size``, which its writer
sets to ``sizeof(struct)``. Fields are only ever appended, so a reader built
against an older header ignores the tail, and a reader built against a newer
header checks ``SPECFEM_HAS_FIELD(ptr, type, field)`` before reading a field
the writer may not know. Arrays of extensible structs are passed as arrays of
pointers so their stride never depends on ``struct_size``. The C++ layer
rejects structs smaller than their version-1 size.

Point groups and material properties
------------------------------------

Points are delivered in groups, one per material system of SPECFEM++, i.e. one
per (medium, property) pair such as elastic/isotropic or elastic/anisotropic.
A plugin therefore sees the same medium x property structure as the solver.
The groups partition the points in order: group ``g`` covers points
``[begin, begin + count)`` of :cpp:struct:`specfem_points`.

Each group carries

- its medium and property tags. Their values are the SPECFEM++ tag
  identifiers and never change; a unit test checks them against the solver.
- its material properties, named after the fields of the SPECFEM++ point
  property container of that system and stored point-major: property ``k`` of
  local point ``p`` is ``properties[p * nproperties + k]``.
- the requested quantities and their buffer layout, as
  :cpp:struct:`specfem_quantity_slot` entries ``{quantity, offset, count}``
  set by SPECFEM++. Component counts depend on the medium, so plugins read
  them instead of deriving them.

Property names
~~~~~~~~~~~~~~

The properties of the three-dimensional systems are:

.. list-table::
    :header-rows: 1
    :widths: 15 20 65

    * - Medium
      - Property
      - Property names, in container order
    * - elastic
      - isotropic
      - ``kappa``, ``mu``, ``rho``
    * - elastic
      - anisotropic
      - ``c11``, ``c12``, ``c13``, ``c14``, ``c15``, ``c16``, ``c22``,
        ``c23``, ``c24``, ``c25``, ``c26``, ``c33``, ``c34``, ``c35``,
        ``c36``, ``c44``, ``c45``, ``c46``, ``c55``, ``c56``, ``c66``,
        ``rho`` (Voigt stiffnesses)
    * - acoustic
      - isotropic
      - ``rho_inverse``, ``kappa``
    * - elastic_spin
      - isotropic_cosserat
      - ``rho``, ``kappa``, ``mu``, ``nu``, ``j``, ``lambda_c``, ``mu_c``,
        ``nu_c``

Always look properties up by name: the order is not part of the ABI, and a
plugin that reads by position breaks if a container gains or reorders fields.
Names also differ between systems (acoustic points have ``rho_inverse``, not
``rho``; anisotropic points have no ``kappa`` or ``mu``). Only stored
properties are sent; derived values such as wave speeds are computed by the
plugin, e.g. :math:`v_p = \sqrt{(\kappa + \tfrac{4}{3}\mu)/\rho}` for
elastic/isotropic points.

Reading properties in C++
~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:class:`specfem::plugin::Points` validates the points and exposes one
:cpp:class:`specfem::plugin::PointGroup` per group. ``property(p, name)``
throws if the group has no such property (reported to SPECFEM++ as an error
naming the property); ``property_index(name)`` returns -1 instead, for
optional properties, and lets hot loops resolve the name once per group.

.. code-block:: cpp

    void setup(const specfem::plugin::Points &points) override {
      using specfem::plugin::Medium;
      using specfem::plugin::Property;

      for (const auto &group : points.groups()) {
        const int i_kappa = group.property_index("kappa"); // -1 if absent

        for (std::int64_t p = 0; p < group.size(); ++p) {
          const std::int64_t i = group.begin() + p; // index into coordinates
          const double x = points.x(i);
          const double z = points.z(i);

          if (group.medium() == Medium::elastic &&
              group.property() == Property::isotropic) {
            const double rho = group.property(p, "rho");
            const double mu = group.property(p, "mu");
            const double kappa = group.property(p, i_kappa);
            // ... check against the plugin's own model, store what fill needs
          } else if (group.medium() == Medium::acoustic) {
            const double rho = 1.0 / group.property(p, "rho_inverse");
            const double kappa = group.property(p, i_kappa);
            // ...
          }
        }
      }
    }

To refuse a material system altogether, return ``false`` from ``supports`` for
it; SPECFEM++ queries ``supports`` before ``setup``.

Reading properties in C
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

    static int find_property(const specfem_point_group *group,
                             const char *name) {
      for (int32_t k = 0; k < group->nproperties; ++k)
        if (strcmp(group->property_names[k], name) == 0)
          return k;
      return -1;
    }

    static specfem_status setup(void *self, const specfem_points *points) {
      for (int32_t g = 0; g < points->ngroups; ++g) {
        const specfem_point_group *group = points->groups[g];
        const int k_rho = find_property(group, "rho");
        for (int64_t p = 0; p < group->count; ++p) {
          const int64_t i = group->begin + p; /* index into coordinates */
          const double rho =
              k_rho >= 0 ? group->properties[p * group->nproperties + k_rho]
                         : 0.0;
          /* points->x[i], points->nz[i], ... */
        }
      }
      return specfem_status_ok;
    }

Filling buffers
~~~~~~~~~~~~~~~

``fill`` receives one buffer per group, in group order. In C++, the SDK
checks every buffer against the shape recorded at ``setup`` before calling the
plugin, and :cpp:func:`specfem::plugin::Buffer::set` throws on an index
outside the buffer. Use the group's quantity slots to find where each
quantity lives:

.. code-block:: cpp

    // In setup: remember where velocity lives in each group (-1 if absent).
    for (const auto &group : points.groups()) {
      const auto *slot = group.slot(specfem::plugin::Quantity::velocity);
      velocity_offset_.push_back(slot ? slot->offset : -1);
    }

    // In fill: buffers[g] belongs to group g.
    for (std::size_t g = 0; g < buffers.size(); ++g)
      for (std::int64_t p = 0; p < buffers[g].number_of_points(); ++p)
        for (int s = 0; s < buffers[g].number_of_steps(); ++s)
          for (int c = 0; c < 3 && velocity_offset_[g] >= 0; ++c)
            buffers[g].set(p, velocity_offset_[g] + c, s,
                           velocity(g, p, c, first_step + s));

C ABI
-----

Everything that crosses the library boundary is defined in
``specfem/plugin/injection_abi.h``. All C names use the ``specfem_`` /
``SPECFEM_`` prefix, matching the exported entry point
``specfem_injection_plugin``.

Function table and data
~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: specfem_injection_plugin_table
    :members:

.. doxygenstruct:: specfem_simulation_info
    :members:

.. doxygenstruct:: specfem_communicator
    :members:

.. doxygenstruct:: specfem_points
    :members:

.. doxygenstruct:: specfem_point_group
    :members:

.. doxygenstruct:: specfem_quantity_slot
    :members:

.. doxygenstruct:: specfem_buffer
    :members:

.. doxygenstruct:: specfem_manifest
    :members:

Types and values
~~~~~~~~~~~~~~~~

The enumerated ABI types are fixed-width integers; their values are listed
below.

.. doxygentypedef:: specfem_medium

.. doxygenenum:: specfem_medium_values

.. doxygentypedef:: specfem_property

.. doxygenenum:: specfem_property_values

.. doxygentypedef:: specfem_quantity

.. doxygenenum:: specfem_quantity_values

.. doxygentypedef:: specfem_status

.. doxygenenum:: specfem_status_values

.. doxygentypedef:: specfem_dtype

.. doxygenenum:: specfem_dtype_values

.. doxygentypedef:: specfem_injection_plugin_entry

.. doxygendefine:: SPECFEM_INJECTION_ABI_VERSION

.. doxygendefine:: SPECFEM_INJECTION_PLUGIN_ENTRY_NAME

Macros
~~~~~~

.. doxygendefine:: SPECFEM_HAS_FIELD

.. doxygendefine:: SPECFEM_DEFINE_INJECTION_MANIFEST

C++ layer
---------

.. doxygendefine:: SPECFEM_DEFINE_INJECTION_PLUGIN

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

.. doxygenenum:: specfem::plugin::Medium

.. doxygenenum:: specfem::plugin::Property

.. doxygenenum:: specfem::plugin::Quantity

.. doxygenfunction:: specfem::plugin::is_valid_medium

.. doxygenfunction:: specfem::plugin::is_valid_property

.. doxygenfunction:: specfem::plugin::is_valid_quantity
