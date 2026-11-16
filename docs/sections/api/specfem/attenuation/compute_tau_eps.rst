``specfem::attenuation::compute_tau_eps``
=========================================

.. doxygenfunction:: specfem::attenuation::compute_tau_eps

The fit itself is :cpp:func:`specfem::attenuation::globe::search_tau_eps`,
the only Q search in the code. This function converts its result to the
1/L normalization and does not clip Q.
