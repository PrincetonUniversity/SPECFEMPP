
.. _compute:

``specfem::compute``
====================

.. doxygennamespace:: specfem::compute
    :desc-only:

``specfem::compute::initialize_mass_matrix``
--------------------------------------------

.. doxygenfunction:: specfem::compute::initialize_mass_matrix

``specfem::compute::update_wavefields``
---------------------------------------

.. doxygenfunction:: specfem::compute::update_wavefields

``specfem::compute::compute_seismograms``
-----------------------------------------

.. doxygenfunction:: specfem::compute::compute_seismograms

``specfem::compute::compute_derivatives``
-----------------------------------------

.. doxygenfunction:: specfem::compute::compute_derivatives

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    compute_mass_matrix
    compute_stiffness_interaction
    stiffness_kernels
    compute_source_interaction
    compute_material_derivatives
    compute_coupling
    invert_mass_matrix
    divide_mass_matrix
    compute_seismogram
