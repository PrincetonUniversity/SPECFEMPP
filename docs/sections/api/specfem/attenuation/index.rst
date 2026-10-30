``specfem::attenuation``
========================

.. doxygennamespace:: specfem::attenuation
    :desc-only:

.. toctree::
    :maxdepth: 1

    tau_sigma.rst
    compute_tau_eps.rst
    compute_factors.rst
    compute_band.rst
    compute_integration_factors.rst
    maxwell.rst
    globe.rst
    optimization.rst

Formulations
------------

2D uses the routines in ``specfem::attenuation``: a 1/L-normalized Maxwell
model, ``one_minus_sum_beta = sum(tau_epsilon / tau_sigma)`` and a 1/L
dispersion factor, fitted with ``NelderMeadSimplex``.

All 3D meshes, globe and non-globe, use ``specfem::attenuation::globe``
(see :doc:`globe`), which follows SPECFEM3D_GLOBE: an unnormalized Maxwell model
fitted in double precision with ``Fminsearch``,
``one_minus_sum_beta = 1 + sum(tau_epsilon / tau_sigma - 1)``, Liu's
reference-to-relaxed scale, and relaxation rates that divide the unrelaxed
modulus by ``one_minus_sum_beta``. Globe has no bulk attenuation; non-globe 3D
isotropic elements apply the same formulas to Q_kappa. Anisotropic elements
attenuate shear only, as in globe, and ignore Q_kappa.

Native non-globe 3D elastic materials, isotropic and anisotropic, reject
``Qkappa <= 0`` or ``Qmu <= 0``. When attenuation is enabled, the native reader
treats a material with both ``Qkappa = 9999`` and ``Qmu = 9999`` as elastic and
skips attenuation fitting. These input conventions are retained independently
of the globe formulation.

For Q values that reach the globe fitting routines, Q at or below zero, above
9000, or infinite becomes 9000. NaN is rejected. Thus an individual Q of 9999
is capped to 9000 unless the native reader selects the elastic material as
described above. As in SPECFEM3D, this includes per-GLL Q values from a
property file.
Q values are grouped by ``max(1, int(Q * 10))`` as in globe's
storage table: the first Q seen in a bucket is fitted, and later Q values are
replaced by it, including in the stored model. Non-globe 3D materials are
bucketed in material order; GLL-varying Q is bucketed per recomputation. The
first Q in a bucket can therefore differ from globe's mesher order when two
different Q values share a bucket.

3D anisotropic elements
-----------------------

Elastic anisotropic elements carry the same per-GLL Q model, stress relaxation
times, and Runge--Kutta memory variables as isotropic elastic elements. Acoustic
outer-core elements have no attenuation container. Anisotropic attenuation is
shear-only, following SPECFEM3D_GLOBE: ``Qkappa`` is stored but does not
attenuate, so the bulk scale stays one and the bulk relaxation rates are zero,
on globe and non-globe meshes alike.

On non-globe (Cartesian) meshes, Cartesian z is the radial axis. The shift
described below for globe crust/mantle elements is applied with an identity
rotation, as at the globe's north pole. Anisotropic materials from the native
reader hold the physical (reference-frequency) stiffnesses. The assembly shifts
them once the property container is filled. Property-file reads on Cartesian
meshes take the same path.

For crust/mantle elements, after the globe model builder has converted Love parameters to Cartesian
stiffnesses, attenuation rotates each tensor into its radial basis, recovers
``A``, ``N``, ``L`` and ``F``, and shifts both radial shear moduli while preserving
``eta = F / (A - 2 L)``. It then rotates all 21 coefficients back to Cartesian
axes. Full anisotropy uses the same projection onto radial Love parameters;
other radial tensor components are preserved.

The two setup stages follow SPECFEM3D_GLOBE. First, the reference-frequency
model is shifted to relaxed moduli with the compliance-based scale from
``globe::get_attenuation_modulus_scales``. Next, the tensor is shifted to unrelaxed
moduli with ``one_minus_sum_beta = 1 + sum(tau_epsilon / tau_sigma - 1)``.
Each mantle stage preserves Love's eta. Exact unity multipliers bypass the rotations.
The isotropic and anisotropic paths use the same Q fit, frequency band and
``tau_sigma``.

The shared ``h_recomputed`` guard prevents a second shift, including through a
copy of the container. The setup-only spherical coordinate cache must be
available for mantle recomputation and can be released afterwards. Inner-core
recomputation uses the unrotated five-coefficient hexagonal update instead of
radial rotations: ``c11``, ``c12``, ``c13``, ``c33``, and ``c44`` are shifted from
``c44`` in both stages, then expanded into all 21 entries. Region metadata is
passed through both globe model setup and property-file reads. Fictitious
central-cube elements are excluded. Property-file reads
supply physical Cartesian coefficients before this step; writes subtract stored
per-coefficient shift increments to recover the physical model.

Globe's current ``prepare_elastic_elements.F90`` assigns the final shifted
radial ``d44`` to ``muvstore``. Accordingly, the memory-variable relaxation
rates use that component, rounded to storage precision:
``2 * d44 * (tau_epsilon / tau_sigma - 1) / tau_sigma / one_minus_sum_beta``.
This includes the final correction to ``factor_common`` in
``prepare_elastic_elements.F90`` (mantle line 536, inner-core line 754). For full
anisotropy this is radial ``c44``, rather than the average ``L = (c44+c55)/2``.
Consequently the memory-variable modulus defect uses the relaxed modulus,
although the retained ``c44`` is unrelaxed.

Reference tests independently fit Q with globe's vendored Fortran simplex and
use the solver's original modulus-factor routines (adapted to physical units).
They compare all 21 shifted stiffnesses and the modulus-weighted memory rates;
no scale computed by SPECFEM++ is supplied to the reference calculation.
