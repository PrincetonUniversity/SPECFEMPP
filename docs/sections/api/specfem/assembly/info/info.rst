
.. _assembly_info:

``specfem::assembly::Info``
===========================

.. doxygenstruct:: specfem::assembly::Info
    :members:


Globe diagnostics
^^^^^^^^^^^^^^^^^

When ``element_types.has_element_context()`` is true, the 3D report adds
statistics for crust/mantle, outer core and inner core: global element counts,
GLL radius bounds, minimum GLL spacing, propagating wave speed bounds and each
region's suggested CFL time step. Cartesian report formatting is unchanged.
Empty regions have zero counts and zero-valued statistics.

``minimum_spacing_location`` and ``cfl_location`` identify the owning rank,
compute element and original database element (zero-based indices local to that
rank). The radius is measured at the first endpoint of the shortest GLL edge
in the limiting element. Ties select the lowest rank, then the lowest compute
element. The CFL estimate uses that element's minimum spacing and maximum wave
speed, which can occur at different GLL points.

``total_volume`` integrates Jacobian times the tensor-product GLL weights;
``total_mass`` additionally multiplies by pointwise density. Both accumulate in
double precision and sum over the active MPI communicator. Reported lengths,
velocities, times, volumes and masses use SI units. All region statistics and
limiting locations are identical on every participating rank.

The optional fourth constructor argument accepts a pointer to existing
assembly Jacobians. The three-argument constructor computes them when globe
context is present. The normal assembly constructor reuses its Jacobians.

.. doxygenstruct:: specfem::assembly::RegionInfo
    :members:

.. doxygenstruct:: specfem::assembly::InfoLocation
    :members:


Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

Bounds
""""""

.. doxygenstruct:: specfem::assembly::info::impl::Bounds
    :members:

BoundingBox
"""""""""""

.. doxygenstruct:: specfem::assembly::info::impl::BoundingBox
    :members:


Computation Functions
"""""""""""""""""""""

.. doxygenfunction:: specfem::assembly::info::impl::compute_average_gll_spacing

.. doxygenfunction:: specfem::assembly::info::impl::compute_minimum_period

.. doxygenfunction:: specfem::assembly::info::impl::compute_suggested_timestep

Distance Computation
""""""""""""""""""""

.. doxygenfunction:: specfem::assembly::info::impl::compute_gll_distances

.. doxygenfunction:: specfem::assembly::info::impl::compute_element_sizes

Scatter-Based Reduction
"""""""""""""""""""""""

.. doxygenstruct:: specfem::assembly::info::impl::LocalMinMax
    :members:

.. doxygenstruct:: specfem::assembly::info::impl::ScatterMinMax
    :members:

.. doxygenstruct:: specfem::assembly::info::impl::ScatterMinMax::Accessor
    :members:
