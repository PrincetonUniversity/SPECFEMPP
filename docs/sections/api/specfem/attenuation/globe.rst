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

These helpers normalize Q values passed to the fitting routines. Native
non-globe 3D material validation and the reader's elastic sentinel are applied
before fitting; see :doc:`index` for these input conventions.

.. doxygenfunction:: specfem::attenuation::globe::effective_quality_factor

.. doxygenfunction:: specfem::attenuation::globe::quality_factor_bucket

.. doxygenclass:: specfem::attenuation::globe::QualityFactorTable
    :members:

Q fit
-----

.. doxygenstruct:: specfem::attenuation::globe::MaxwellFactors
    :members:

.. doxygenfunction:: specfem::attenuation::globe::maxwell

.. doxygenstruct:: specfem::attenuation::globe::AttenuationObjective
    :members:

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
