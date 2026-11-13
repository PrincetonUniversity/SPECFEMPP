SPECFEM3D_GLOBE formulation (3D)
================================

Every 3D mesh, globe and non-globe, uses these routines. They port the Q fit
in SPECFEM3D_GLOBE's ``src/meshfem3D/model_attenuation.f90`` and the modulus
factors in ``src/specfem3D/get_attenuation.f90``. 2D keeps the 1/L-normalized
routines documented on the other pages.

.. doxygennamespace:: specfem::attenuation::globe
    :desc-only:

Quality factor
--------------

Globe clips Q before fitting and before the dispersion correction. Native
non-globe 3D material validation and the reader's elastic sentinel are applied
before fitting; see :doc:`index` for these input conventions. Q is bucketed
with the formulation-independent :doc:`quality_factor_table`.

.. doxygenfunction:: specfem::attenuation::globe::clip_quality_factor

.. doxygenfunction:: specfem::attenuation::globe::make_quality_factor_table

Q fit
-----

.. doxygenstruct:: specfem::attenuation::globe::MaxwellFactors
    :members:

.. doxygenfunction:: specfem::attenuation::globe::maxwell

.. doxygenstruct:: specfem::attenuation::globe::AttenuationObjective
    :members:

.. note::

   Convergence is stricter than SPECFEM3D_GLOBE by design. Globe's
   ``fminsearch`` only aborts when the evaluation count exceeds its budget,
   so a fit that stops on the iteration budget without meeting the
   tolerances is used silently. ``search_tau_eps`` throws for every fit
   that does not meet the tolerances. A Q that globe accepts with a
   budget-limited fit therefore raises an error here instead of running with
   an inaccurate attenuation model. Converged fits match globe.

.. doxygenfunction:: specfem::attenuation::globe::search_tau_eps

.. doxygenfunction:: specfem::attenuation::globe::compute_tau_eps

Modulus factors
---------------

.. doxygenfunction:: specfem::attenuation::globe::get_attenuation_property_values

.. doxygenstruct:: specfem::attenuation::globe::ModulusScales
    :members:

.. doxygenfunction:: specfem::attenuation::globe::get_attenuation_modulus_scales

.. doxygenstruct:: specfem::attenuation::globe::QualityFactorFit
    :members:

.. doxygenfunction:: specfem::attenuation::globe::fit_quality_factor
