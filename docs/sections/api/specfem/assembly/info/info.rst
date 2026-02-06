
.. _assembly_info:

``specfem::assembly::Info``
===========================

.. doxygenstruct:: specfem::assembly::Info
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
