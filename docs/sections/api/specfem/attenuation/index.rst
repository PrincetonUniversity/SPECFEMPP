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
    quality_factor_table.rst
    optimization.rst

Formulation
-----------

2D, 3D Cartesian and 3D globe meshes share one formulation, that of Chapter 5
of Tromp's attenuation notes (sections 5.4 to 5.6). With :math:`L` =
``N_SLS`` mechanisms:

- **Solid** (eq. 5.33). A generalized Maxwell solid normalized by
  :math:`1/L`,
  :math:`M(\omega) = M_R\,\frac{1}{L}\sum_i
  \frac{1 + i\omega\tau_{\epsilon_i}}{1 + i\omega\tau_{\sigma_i}}`
  (:doc:`maxwell`). ``tau_sigma`` is log-spaced over the band and
  ``tau_epsilon`` is fitted to a constant Q over the band by
  :doc:`compute_tau_eps`.
- **Unrelaxed modulus and defects** (eqs. 5.34 to 5.36, 5.58, 5.59, 5.62,
  5.63). The mechanism defects are :math:`\delta M_i = M_R\,\beta_i` with
  :math:`\beta_i = (\tau_{\epsilon_i}/\tau_{\sigma_i} - 1)/L`, and
  :math:`M_U/M_R = 1 + \sum_i \beta_i = \frac{1}{L}\sum_i
  \tau_{\epsilon_i}/\tau_{\sigma_i}`. They are stored as ``beta`` and
  ``unrelaxed_ratio`` in ``AttenuationPropertyValues``.
- **Memory variables** (eqs. 5.64 and 5.65). Memory variable :math:`i` decays
  with :math:`\tau_{\sigma_i}` and is driven by :math:`\mathrm{tr}(\varepsilon)`
  or :math:`\mathbf{d}` with coefficient :math:`\delta M_i/\tau_{\sigma_i}
  = M_R\,\beta_i/\tau_{\sigma_i}`, where :math:`M_R = M_U/(M_U/M_R)` is
  recovered from the stored unrelaxed modulus
  (``AttenuationPropertyValues::relaxation_rate``). The solver subtracts the
  sum of the memory variables from the stress, as in (5.57), and stores the
  shear memory variable as :math:`2\mathbf{r}_\mu` (shear coefficient
  :math:`2\,\delta\mu_i/\tau_{\sigma_i}`). This places the factor 2 that
  substituting (5.50) into (5.48) produces,
  :math:`2\mu_U\mathbf{d} - 2\sum_\ell \mathbf{r}^\ell_\mu`, which (5.57)
  as printed omits.
- **Modulus scaling** (eqs. 5.70 to 5.76). Model moduli are given at the
  reference frequency :math:`f_0`. :math:`M_R` is chosen so that the storage
  modulus at the band center matches the absorption band,
  :math:`M_1(\omega_c) = M_0\,[1 + 2\ln(f_c/f_0)/(\pi Q)]`.
  ``get_attenuation_modulus_scales`` returns the two factors separately:
  ``relaxed`` :math:`= M_R/M_0` and ``ratio`` :math:`= M_U/M_R`. Isotropic
  moduli are scaled by their product; anisotropic tensors use them in two
  stages (below).

``fit_quality_factor`` runs the fit, the coefficients and the scales for one
Q. Whether kappa attenuates is decided by the mesh's ``bulk_attenuation`` flag
alone, for isotropic and anisotropic elements alike. Globe meshes turn it off.

Quality factors
---------------

Native non-globe 3D elastic materials, isotropic and anisotropic, reject
``Qkappa <= 0`` or ``Qmu <= 0``. When attenuation is enabled, the native reader
treats a material with both ``Qkappa = 9999`` and ``Qmu = 9999`` as elastic and
skips attenuation fitting.

In 3D, as in the SPECFEM3D and SPECFEM3D_GLOBE meshers, any other Q at or
below zero, above 9000, or infinite becomes 9000. This includes per-GLL Q
values from a property file. NaN is rejected. 2D uses Q as given.

Q values are grouped by ``max(1, int(Q * 10))`` with
:doc:`quality_factor_table`: the first Q seen in a bucket is fitted, and later
Q values are replaced by it, including in the stored model. Non-globe 3D
materials are bucketed in material order; GLL-varying Q is bucketed per
recomputation. The first Q in a bucket can therefore differ from globe's
mesher order when two different Q values share a bucket.

Assumptions beyond the isotropic theory
---------------------------------------

The notes treat an isotropic solid with one shear and one bulk Q. Elastic
anisotropic elements carry the same per-GLL ``Qmu`` and ``Qkappa``, stress
relaxation times and Runge--Kutta memory variables as isotropic ones, and the
following is assumed on top of the chapter.

**Shear: Love-parameter shift in a radial frame.** Each tensor is rotated into
a frame whose third axis is the symmetry axis. There the Love parameters
``A``, ``N``, ``L`` and ``F`` are recovered and both shear moduli ``L`` and
``N`` are scaled, preserving ``eta = F / (A - 2 L)``. All 21 coefficients are
then rotated back. Full anisotropy uses the same projection onto radial Love
parameters; other radial components are preserved. The scaling uses the
chapter's two factors in turn, as SPECFEM3D_GLOBE's ``prepare_attenuation.f90``
and ``prepare_elastic_elements.F90`` do: ``relaxed`` takes the reference
tensor to the relaxed one, with the coefficients rounded to storage precision
in between, and ``ratio`` takes it to the unrelaxed one. For a nonzero shift,
an undefined or nonfinite eta (including ``A = 2 L``) raises an error before
the tensor is updated. Exact unity factors bypass the rotations.

**Symmetry axis.**

- Globe crust and mantle: the local radial direction.
- Globe inner core: no rotation. The five hexagonal coefficients ``c11``,
  ``c12``, ``c13``, ``c33`` and ``c44`` are shifted from ``c44`` in both
  stages, then expanded into all 21 entries. Fictitious central-cube elements
  are excluded.
- Cartesian (non-globe) meshes: Cartesian z, the globe's frame at the north
  pole.

**Shear modulus of the memory variables.** The shear relaxation rates use the
final shifted radial ``c44``, rounded to storage precision, as
SPECFEM3D_GLOBE's ``muvstore`` does. For full anisotropy this is radial
``c44`` rather than ``L = (c44 + c55) / 2``.

**Bulk: Voigt-kappa increment.** With bulk attenuation enabled (non-globe
meshes), the bulk modulus is shifted by the isotropic increment
``kappa_V * (s - 1)`` added to ``c11, c22, c33, c12, c13, c23``, where
``kappa_V`` is the Voigt bulk modulus of the reference tensor and ``s`` the
bulk multiplier ``relaxed * ratio``. The increment is frame-independent and
reproduces ``kappa <- s * kappa`` for isotropic input. The bulk relaxation
rates use the unrelaxed ``s * kappa_V``.

Lifecycle
---------

Anisotropic materials from the native reader, globe model setup and
property-file reads all supply physical (reference-frequency) stiffnesses. The
assembly shifts them once the property container is filled; writes subtract
the stored per-coefficient shift increments to recover the physical model.
The shared ``h_recomputed`` guard prevents a second shift, including through a
copy of the container. The setup-only spherical coordinate cache must be
available for globe mantle recomputation and can be released afterwards.
Acoustic outer-core elements have no attenuation container.

The tensor shift is tested against SPECFEM3D_GLOBE's vendored Fortran rotations
and update formulas, given the fit from ``fit_quality_factor``. The fit itself
is tested against the formulation above: flat Q over the band,
:math:`M_1(\omega_c)` matching the absorption band, and :math:`M_U/M_R` as the
high-frequency limit.
