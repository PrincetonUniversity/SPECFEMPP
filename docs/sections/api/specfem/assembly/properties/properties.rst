
.. _assembly_properties:

``specfem::assembly::properties``
=================================

.. doxygenstruct:: specfem::assembly::properties
    :members:

Dimension-Specific Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: specfem::assembly::properties< specfem::element::dimension_tag::dim2 >
    :members:

.. doxygenstruct:: specfem::assembly::properties< specfem::element::dimension_tag::dim3 >
    :members:

Data Access Functions
^^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: PropertiesDataAccess
    :content-only:

Globe model initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Globe assemblies allocate the existing property containers without reading a
Cartesian material table. When no GLL property reader is supplied,
``read_globe_properties`` fills them using the model configuration retained in
the thin database. Sampling uses interpolated **reference** coordinates and
compute-ordered element context, including region, radial shell and crust flags.

The oracle owns the Voigt averaging convention. Elastic isotropic properties are
``rho``, ``mu = rho * vs_iso**2`` and
``kappa = rho * vp_iso**2 - 4/3 * mu``. Acoustic properties are
``rho_inverse = 1/rho`` and ``kappa = rho * vp_iso**2``. Anisotropic elastic
elements receive density and all 21 Cartesian stiffness coefficients in the
container's Voigt order; they are never replaced by isotropic averages.
Attenuation has no anisotropic container yet, so enabling attenuation while any
element is anisotropic is rejected rather than silently running those elements
elastically.

The host loop calls ``ModelEvaluator::evaluate_element`` once per element. Each
batch evaluates its GLL points serially and preserves the catalog's element-level
Moho and sediment state. Properties are copied to device once at the end. A per-rank
debug-level log reports wall time (including oracle initialization and the device copy),
oracle element calls and evaluated GLL points.

A configured property reader takes precedence: the oracle is neither initialized
nor called, and its database validation is bypassed.

.. doxygenfunction:: specfem::assembly::dim3_impl::read_globe_properties

Validation uses ``GlobalSmallMesh``, the replacement for ``single_chunk_1D``,
with all three regions and ellipticity enabled. The globe property tests compare
host and device storage and sampled oracle values, including an anisotropic PREM
variant in which only crust-mantle elements are anisotropic (anisotropic inner
core is not exercised). Set ``SPECFEM_GLOBE_PROFILE`` to a CSV output path when running
``globe_properties_tests`` to export isotropic assembly and direct PREM radial
profiles. Double-precision builds use a relative tolerance of ``1e-10``;
single-precision builds use ``2e-6``.
