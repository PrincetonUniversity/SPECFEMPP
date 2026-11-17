``specfem::attenuation::compute_tau_eps``
=========================================

.. doxygenstruct:: specfem::attenuation::AttenuationObjective
    :members:

.. doxygenfunction:: specfem::attenuation::compute_tau_eps

This is the only Q search in the code, used by every dimension. It does not
clip Q; 3D callers limit Q through :doc:`quality_factor_table`.
