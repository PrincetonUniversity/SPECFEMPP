#include "specfem/injection/fk/impl/driver.hpp"

#include "specfem/injection/fk/impl/field_recovery.hpp"
#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/injection/fk/impl/halfspace_solve.hpp"
#include "specfem/injection/fk/impl/layer_operators.hpp"
#include "specfem/injection/fk/impl/medium_interface_coupler.hpp"
#include "specfem/injection/fk/time_window.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include "specfem/utilities/cubic_bspline.hpp"
#include "specfem/utilities/fft.hpp"

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Identity-initialise a ComplexMatrix<4> (KOKKOS_INLINE_FUNCTION).
// Placed in specfem::injection::fk_impl to follow project namespace rules.
// ---------------------------------------------------------------------------
// set_identity is a KOKKOS_INLINE_FUNCTION defined here (not in a .hpp) to
// avoid exposing a private helper in the public impl headers.  Per project
// convention, definitions in .cpp files use fully-qualified names outside any
// namespace block.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void specfem::injection::fk_impl::set_identity(
    specfem::utilities::ComplexMatrix<4> &M) {
  for (int k = 0; k < 16; ++k)
    M.data[k] = Kokkos::complex<double>(0.0, 0.0);
  M(0, 0) = Kokkos::complex<double>(1.0, 0.0);
  M(1, 1) = Kokkos::complex<double>(1.0, 0.0);
  M(2, 2) = Kokkos::complex<double>(1.0, 0.0);
  M(3, 3) = Kokkos::complex<double>(1.0, 0.0);
}

// ---------------------------------------------------------------------------
// run_fk_driver
// ---------------------------------------------------------------------------

specfem::injection::fk::FkResult specfem::injection::fk_impl::run_fk_driver(
    const specfem::injection::fk::LayeredModel &model,
    const specfem::injection::fk::IncidentWave &wave,
    const specfem::injection::fk::TimeWindow &window,
    const specfem::injection::fk::EvalPoints &points, bool compute_traction,
    specfem::injection::fk::field_derivative derivative) {

  using Cplx = Kokkos::complex<double>;

  // (i*om)^deriv_order applied to the vector field components before the IFFT:
  // 0 = displacement, 1 = velocity, 2 = acceleration.
  const int deriv_order =
      (derivative == specfem::injection::fk::field_derivative::velocity)
          ? 1
          : (derivative ==
                     specfem::injection::fk::field_derivative::acceleration
                 ? 2
                 : 0);

  // =========================================================================
  // Step 1: working-array sizes  (ref fk_core.cpp lines 27–41)
  // =========================================================================
  specfem::injection::fk::FkSizes sizes =
      specfem::injection::fk::compute_fk_sizes(window);

  const int nf_stored = sizes.number_of_stored_frequencies; // NF_FOR_STORING
  const int nf2 = nf_stored + 1; // positive-frequency sample count (incl. DC)
  const int nf = 2 * nf_stored;  // total FFT length

  // npow = log2(nf)  (nf is a power of two)
  int npts2 = nf;
  int npow = 0;
  {
    int tmp = npts2;
    while (tmp > 1) {
      tmp >>= 1;
      ++npow;
    }
  }
  npts2 = 1 << npow; // == nf after the round-up (already a power of two)

  const double df = static_cast<double>(sizes.frequency_step);
  const double dt_fk = 1.0 / (df * static_cast<double>(npts2 - 1));
  // NOTE: this SPECFEM3D FK reference uses a real angular frequency (no
  // complex-frequency / exponential-window damping). sigma0 = 0 reproduces it
  // exactly; a nonzero value would be the Westervelt-style stabilization.
  const double sigma0 = 0.0;

  // t0 = -origin_time  (sign: positive origin_time → wavefront arrives late,
  // t0 negative → time axis starts before arrival).
  const double t0 = -static_cast<double>(wave.origin_time);
  const int nn =
      static_cast<int>(-t0 / dt_fk); // nn >= 0 for positive origin_time

  const double phi = static_cast<double>(wave.azimuth_phi);

  // Half-space record
  const auto &halfspace = model.elastic_layers().back();
  const double hs_velocity =
      (wave.type == specfem::injection::fk::incident_wave_type::p)
          ? static_cast<double>(halfspace.p_velocity)
          : static_cast<double>(halfspace.s_velocity);

  const type_real ray_p_tr =
      wave.ray_parameter(static_cast<type_real>(hs_velocity));
  const double ray_p = static_cast<double>(ray_p_tr);

  // Incident amplitudes (omega-independent, computed on host and captured)
  const specfem::injection::fk_impl::IncidentAmplitude inc_amp =
      specfem::injection::fk_impl::incident_amplitude(halfspace, wave.type,
                                                      wave.amplitude, ray_p_tr);
  const Cplx c1 = inc_amp.c1;
  const Cplx c3 = inc_amp.c3;

  // =========================================================================
  // Step 2: host FFT tables
  // =========================================================================
  specfem::utilities::fft::FftTables fft_tables =
      specfem::utilities::fft::make_fft_tables(npow, -1.0);

  // =========================================================================
  // Step 3: stf_apod and omvec
  // =========================================================================
  const double Tg = static_cast<double>(wave.gaussian_half_duration);

  Kokkos::View<double *, Kokkos::HostSpace> h_omvec("h_omvec", nf2);
  Kokkos::View<double *, Kokkos::HostSpace> h_stf_apod("h_stf_apod", nf2);
  for (int ii = 0; ii < nf2; ++ii) {
    h_omvec(ii) =
        2.0 * specfem::injection::fk_impl::pi * static_cast<double>(ii) * df;
    const double x = h_omvec(ii) * Tg / 2.0;
    h_stf_apod(ii) = std::exp(-x * x);
  }
  Kokkos::View<double *> d_omvec("d_omvec", nf2);
  Kokkos::View<double *> d_stf_apod("d_stf_apod", nf2);
  Kokkos::deep_copy(d_omvec, h_omvec);
  Kokkos::deep_copy(d_stf_apod, h_stf_apod);

  // =========================================================================
  // Step 4: half-space E matrix (omega-independent)
  // =========================================================================
  const specfem::utilities::ComplexMatrix<4> e_mat_host =
      specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(halfspace,
                                                                 ray_p_tr);

  Kokkos::View<Cplx *, Kokkos::HostSpace> h_e_mat("h_e_mat", 16);
  for (int k = 0; k < 16; ++k)
    h_e_mat(k) = e_mat_host.data[k];
  Kokkos::View<Cplx *> d_e_mat("d_e_mat", 16);
  Kokkos::deep_copy(d_e_mat, h_e_mat);

  // =========================================================================
  // Step 5: layer data to device
  // =========================================================================
  const int n_fluid = model.number_of_fluid_layers();
  const int n_elastic = model.number_of_elastic_layers();
  const bool has_fluid = model.has_fluid_layer();

  Kokkos::View<type_real *, Kokkos::HostSpace> h_el_vp("h_el_vp", n_elastic),
      h_el_vs("h_el_vs", n_elastic), h_el_rho("h_el_rho", n_elastic),
      h_el_H("h_el_H", n_elastic);
  for (int i = 0; i < n_elastic; ++i) {
    const auto &el = model.elastic_layers()[i];
    h_el_vp(i) = el.p_velocity;
    h_el_vs(i) = el.s_velocity;
    h_el_rho(i) = el.density;
    h_el_H(i) = el.thickness;
  }
  Kokkos::View<type_real *> d_el_vp("d_el_vp", n_elastic),
      d_el_vs("d_el_vs", n_elastic), d_el_rho("d_el_rho", n_elastic),
      d_el_H("d_el_H", n_elastic);
  Kokkos::deep_copy(d_el_vp, h_el_vp);
  Kokkos::deep_copy(d_el_vs, h_el_vs);
  Kokkos::deep_copy(d_el_rho, h_el_rho);
  Kokkos::deep_copy(d_el_H, h_el_H);

  // Fluid layers: allocate at least size 1 to avoid zero-extent Views.
  const int nf_alloc = (n_fluid > 0) ? n_fluid : 1;
  Kokkos::View<type_real *, Kokkos::HostSpace> h_ac_vp("h_ac_vp", nf_alloc),
      h_ac_rho("h_ac_rho", nf_alloc), h_ac_H("h_ac_H", nf_alloc);
  for (int i = 0; i < n_fluid; ++i) {
    const auto &fl = model.fluid_layers()[i];
    h_ac_vp(i) = fl.p_velocity;
    h_ac_rho(i) = fl.density;
    h_ac_H(i) = fl.thickness;
  }
  Kokkos::View<type_real *> d_ac_vp("d_ac_vp", nf_alloc),
      d_ac_rho("d_ac_rho", nf_alloc), d_ac_H("d_ac_H", nf_alloc);
  Kokkos::deep_copy(d_ac_vp, h_ac_vp);
  Kokkos::deep_copy(d_ac_rho, h_ac_rho);
  Kokkos::deep_copy(d_ac_H, h_ac_H);

  // =========================================================================
  // Step 6: per-frequency storage
  //
  // el_chain[ii, L, 0..15] : cumulative elastic propagator (ref
  // el_chain[ii][L]).
  //   L=n_elastic-1 → identity (half-space slot).
  //   L < n_elastic-1 → P_{L} * gamma0_{L} * el_chain[ii, L+1].
  // ac_chain[ii, L, 0..15] : cumulative acoustic propagator (ref
  // ac_chain[ii][L]).
  //   L=n_fluid-1   → identity.
  //   L < n_fluid-1  → Q_{L+1} * ac_chain[ii, L+1].
  // coeffs[ii, 0..1] : half-space reflection coefficients.
  // =========================================================================
  const int el_chain_size = nf2 * n_elastic * 16;
  const int ac_chain_size = has_fluid ? nf2 * n_fluid * 16 : 1;

  Kokkos::View<Cplx *> d_el_chain("d_el_chain", el_chain_size);
  Kokkos::View<Cplx *> d_ac_chain("d_ac_chain", ac_chain_size);
  Kokkos::View<Cplx *> d_coeffs("d_coeffs", nf2 * 2);

  // =========================================================================
  // Kernel 1: per-frequency propagator chains and half-space coefficients
  // =========================================================================
  {
    const int nf2_k = nf2;
    const int n_elastic_k = n_elastic;
    const int n_fluid_k = n_fluid;
    const bool has_fluid_k = has_fluid;
    const double sigma0_k = sigma0;
    const double ray_p_k = ray_p;
    const Cplx c1_k = c1;
    const Cplx c3_k = c3;
    const specfem::injection::fk::incident_wave_type wave_type = wave.type;

    Kokkos::parallel_for(
        "fk_kernel1", Kokkos::RangePolicy<>(0, nf2_k), KOKKOS_LAMBDA(int ii) {
          const double om_real = d_omvec(ii);
          const Cplx om(om_real, -sigma0_k);

          // Reconstruct E matrix.
          specfem::utilities::ComplexMatrix<4> e_mat;
          for (int k = 0; k < 16; ++k)
            e_mat.data[k] = d_e_mat(k);

          // ---- elastic chain (bottom-up) ----
          // Slot ii*n_elastic_k*16 + (n_elastic_k-1)*16 gets identity.
          const int base_el = ii * n_elastic_k * 16;

          specfem::utilities::ComplexMatrix<4> C;
          specfem::injection::fk_impl::set_identity(C);
          for (int k = 0; k < 16; ++k)
            d_el_chain(base_el + (n_elastic_k - 1) * 16 + k) = C.data[k];

          // Walk L = n_elastic-2 down to 0.
          // The propagator at step L uses the material of layer L
          // (the layer just above the product so far).
          for (int L = n_elastic_k - 2; L >= 0; --L) {
            specfem::injection::fk::ElasticIsotropicLayer el_layer;
            el_layer.p_velocity = d_el_vp(L);
            el_layer.s_velocity = d_el_vs(L);
            el_layer.density = d_el_rho(L);
            el_layer.thickness = d_el_H(L);

            const specfem::utilities::ComplexMatrix<4> P =
                specfem::injection::fk_impl::layer_propagator(
                    el_layer, om, el_layer.thickness,
                    static_cast<type_real>(ray_p_k));

            const double g0 = specfem::injection::fk_impl::elastic_gamma0(
                el_layer, static_cast<type_real>(ray_p_k));

            const specfem::utilities::ComplexMatrix<4> PC = P * C;
            for (int k = 0; k < 16; ++k)
              C.data[k] = PC.data[k] * Cplx(g0, 0.0);

            for (int k = 0; k < 16; ++k)
              d_el_chain(base_el + L * 16 + k) = C.data[k];
          }

          // ---- acoustic chain (bottom-up) ----
          if (has_fluid_k) {
            const int base_ac = ii * n_fluid_k * 16;

            specfem::utilities::ComplexMatrix<4> C2;
            specfem::injection::fk_impl::set_identity(C2);
            for (int k = 0; k < 16; ++k)
              d_ac_chain(base_ac + (n_fluid_k - 1) * 16 + k) = C2.data[k];

            // Walk L = n_fluid-2 down to 0.
            // Step L uses the propagator of layer L+1 (0-based acoustic index).
            for (int L = n_fluid_k - 2; L >= 0; --L) {
              specfem::injection::fk::AcousticLayer ac_layer;
              ac_layer.p_velocity = d_ac_vp(L + 1);
              ac_layer.density = d_ac_rho(L + 1);
              ac_layer.thickness = d_ac_H(L + 1);

              const specfem::utilities::ComplexMatrix<4> Q =
                  specfem::injection::fk_impl::layer_propagator(
                      ac_layer, om, ac_layer.thickness,
                      static_cast<type_real>(ray_p_k));

              C2 = Q * C2;
              for (int k = 0; k < 16; ++k)
                d_ac_chain(base_ac + L * 16 + k) = C2.data[k];
            }
          }

          // ---- N_mat = (elastic chain L=0) * E   or   E if only half-space
          // ---- el_chain[ii, 0] = product of all finite elastic layers
          // (L=0..n_el-2) applied cumulative bottom-up, which when
          // right-multiplied by E gives N.
          specfem::utilities::ComplexMatrix<4> n_mat;
          if (n_elastic_k > 1) {
            specfem::utilities::ComplexMatrix<4> chain0;
            for (int k = 0; k < 16; ++k)
              chain0.data[k] = d_el_chain(base_el + 0 * 16 + k);
            n_mat = chain0 * e_mat;
          } else {
            n_mat = e_mat;
          }

          // ---- Qmat_full = Q_0 * ac_chain[0] ----
          specfem::utilities::ComplexMatrix<4> qmat_full;
          specfem::injection::fk_impl::set_identity(
              qmat_full); // value only used when has_fluid_k

          if (has_fluid_k) {
            specfem::injection::fk::AcousticLayer ac0;
            ac0.p_velocity = d_ac_vp(0);
            ac0.density = d_ac_rho(0);
            ac0.thickness = d_ac_H(0);

            const specfem::utilities::ComplexMatrix<4> Q0 =
                specfem::injection::fk_impl::layer_propagator(
                    ac0, om, ac0.thickness, static_cast<type_real>(ray_p_k));

            const int base_ac = ii * n_fluid_k * 16;
            specfem::utilities::ComplexMatrix<4> ac_chain0;
            for (int k = 0; k < 16; ++k)
              ac_chain0.data[k] = d_ac_chain(base_ac + 0 * 16 + k);

            qmat_full = Q0 * ac_chain0;
          }

          // ---- half-space coefficients ----
          const auto coeffs_arr =
              specfem::injection::fk_impl::halfspace_coefficients(
                  n_mat, has_fluid_k, qmat_full, wave_type, c1_k, c3_k);

          d_coeffs(ii * 2 + 0) = coeffs_arr[0];
          d_coeffs(ii * 2 + 1) = coeffs_arr[1];
        });

    Kokkos::fence();
  }

  // =========================================================================
  // Step 7: allocate FkResult and per-point scratch Views
  // =========================================================================
  const int npoints = points.size();
  const bool needs_pressure = has_fluid;
  // Traction is meaningful only for elastic points; always allocate it when
  // compute_traction is true (the kernel only writes non-zero values for
  // elastic points).
  specfem::injection::fk::FkResult result(npoints, nf_stored, compute_traction,
                                          needs_pressure);

  // Per-point frequency-domain scratch: [npoints, 5, nf] complex
  // Per-point time-domain scratch: [npoints, 5, npts2] double
  Kokkos::View<Cplx ***> d_field_f("d_field_f", npoints, 5, nf);
  Kokkos::View<double ***> d_field("d_field", npoints, 5, npts2);
  // Spline scratch: [npoints, npts2] double — reused for each component.
  // Avoids stack-allocated VLAs inside the GPU kernel.
  Kokkos::View<double **> d_col("d_col", npoints, npts2);
  Kokkos::View<double **> d_cvals("d_cvals", npoints, npts2);

  // =========================================================================
  // Kernel 2: per-point field recovery, IFFT, taper, spline
  // =========================================================================
  {
    const int nf2_k = nf2;
    const int nf_k = nf;
    const int npts2_k = npts2;
    const int nf_stored_k = nf_stored;
    const int npow_k = npow;
    const int n_elastic_k = n_elastic;
    const int n_fluid_k = n_fluid;
    const bool has_fluid_k = has_fluid;
    const double sigma0_k = sigma0;
    const double dt_fk_k = dt_fk;
    // The inverse FFT normalization uses the SEM time step (ref passes `dt`,
    // not dt_fk, to FFTinv); only the exponential taper below uses dt_fk.
    const double dt_sem_k = static_cast<double>(window.dt);
    const double t0_k = t0;
    const double ray_p_k = ray_p;
    const double phi_k = phi;
    const int nn_k = nn;
    const double origin_x = static_cast<double>(wave.origin_x);
    const double origin_y = static_cast<double>(wave.origin_y);
    const double origin_z = static_cast<double>(wave.origin_z);
    const specfem::injection::fk::incident_wave_type wave_type = wave.type;
    const int deriv_order_k = deriv_order;
    const bool compute_traction_k = compute_traction;
    const Cplx c1_k = c1;
    const Cplx c3_k = c3;
    const double vp_hs_k = static_cast<double>(halfspace.p_velocity);

    const int *bit_reversal_ptr = fft_tables.bit_reversal.data();
    const Cplx *twiddles_ptr = fft_tables.twiddles.data();

    auto d_displacement = result.displacement();
    auto d_traction = result.traction();
    auto d_pressure = result.pressure();
    auto d_time_delays = result.time_delays();
    // Capture spline scratch Views (device-resident, avoids stack arrays).
    auto d_col_capture = d_col;
    auto d_cvals_capture = d_cvals;

    Kokkos::parallel_for(
        "fk_kernel2", Kokkos::RangePolicy<>(0, npoints),
        KOKKOS_LAMBDA(int ipt) {
          // -----------------------------------------------------------------
          // Time delay (ref lines 339-343)
          // eta_p = sqrt(1/vp^2 - p^2), real part (upward-going P vertical
          // slowness).  Tdelay = p*(x-x0)*cos(phi) + p*(y-y0)*sin(phi)
          //                     + eta_p*(0 - z0)
          // The reference sets the half-space top at z==0; z0 < 0 means the
          // reference point is above the half-space (within the layered model).
          // -----------------------------------------------------------------
          const double eta_p_sq = 1.0 / (vp_hs_k * vp_hs_k) - ray_p_k * ray_p_k;
          const double eta_p_real =
              (eta_p_sq >= 0.0) ? Kokkos::sqrt(eta_p_sq) : 0.0;

          const double xi = static_cast<double>(points.x(ipt));
          const double yi = static_cast<double>(points.y(ipt));
          const double zi = static_cast<double>(points.z(ipt));

          const double Tdelay = ray_p_k * (xi - origin_x) * Kokkos::cos(phi_k) +
                                ray_p_k * (yi - origin_y) * Kokkos::sin(phi_k) +
                                eta_p_real * (0.0 - origin_z);

          d_time_delays(ipt) = static_cast<type_real>(Tdelay);

          // -----------------------------------------------------------------
          // Classify point as elastic or acoustic (ref lines 119-141)
          // -----------------------------------------------------------------
          bool is_elastic = true;
          if (points.elastic_override.extent(0) > 0) {
            const char ov = points.elastic_override(ipt);
            if (ov == 'A')
              is_elastic = false;
            else if (ov == 'E')
              is_elastic = true;
            else {
              // Depth-based classification: fluid layers are at the top.
              if (has_fluid_k) {
                double fluid_total = 0.0;
                for (int f = 0; f < n_fluid_k; ++f)
                  fluid_total += static_cast<double>(d_ac_H(f));
                is_elastic = (zi >= fluid_total);
              }
            }
          } else if (has_fluid_k) {
            double fluid_total = 0.0;
            for (int f = 0; f < n_fluid_k; ++f)
              fluid_total += static_cast<double>(d_ac_H(f));
            is_elastic = (zi >= fluid_total);
          }

          // -----------------------------------------------------------------
          // Compute zz (height above half-space top, ref sign convention).
          // halfspace top in model z = fluid_total + elastic_finite_total.
          // zz > 0 means within the finite layers; zz <= 0 in the half-space.
          // -----------------------------------------------------------------
          double fluid_total_k = 0.0;
          for (int f = 0; f < n_fluid_k; ++f)
            fluid_total_k += static_cast<double>(d_ac_H(f));

          double elastic_finite_total = 0.0;
          for (int e = 0; e < n_elastic_k - 1; ++e)
            elastic_finite_total += static_cast<double>(d_el_H(e));

          const double halfspace_top_z = fluid_total_k + elastic_finite_total;
          const double zz = halfspace_top_z - zi; // positive = above hs top

          const bool in_halfspace = is_elastic && (zz <= 0.0);

          // -----------------------------------------------------------------
          // Layer determination (ref lines 356-403)
          // -----------------------------------------------------------------
          int ilayer_el = n_elastic_k - 1; // default = half-space slot
          int ilayer_ac = 0;
          double height = 0.0;

          if (!in_halfspace) {
            if (is_elastic) {
              // Elastic layer walk: find elastic layer containing point.
              // zi_el = depth into the elastic block (from its top).
              const double zi_el = zi - fluid_total_k;
              double s_accum = 0.0;
              ilayer_el = n_elastic_k - 1;
              for (int j = 0; j < n_elastic_k - 1; ++j) {
                s_accum += static_cast<double>(d_el_H(j));
                if (zi_el <= s_accum) {
                  ilayer_el = j;
                  break;
                }
              }
              // height = distance of point above the bottom of ilayer_el
              // (ref: height = zz - s, where s = sum of thicknesses of layers
              // above ilayer_el within the finite elastic block, measured
              // upward from half-space top).
              double s_above = 0.0;
              for (int k = ilayer_el + 1; k < n_elastic_k - 1; ++k)
                s_above += static_cast<double>(d_el_H(k));
              height = zz - s_above;
            } else {
              // Acoustic layer walk: find fluid layer containing point.
              double s_accum = 0.0;
              ilayer_ac = 0;
              for (int j = 0; j < n_fluid_k; ++j) {
                const double layer_bottom =
                    s_accum + static_cast<double>(d_ac_H(j));
                if (zi <= layer_bottom) {
                  ilayer_ac = j;
                  break;
                }
                s_accum += static_cast<double>(d_ac_H(j));
              }
              // height = distance above the bottom of ilayer_ac (within the
              // fluid block; bottom = larger zi value, upward = smaller zi).
              double s_below = 0.0; // thickness of fluid layers below ilayer_ac
              for (int k = ilayer_ac + 1; k < n_fluid_k; ++k)
                s_below += static_cast<double>(d_ac_H(k));
              // Bottom of ilayer_ac in global z = fluid_total - s_below
              const double layer_bottom_z = fluid_total_k - s_below;
              height = layer_bottom_z - zi;
            }
          }

          // -----------------------------------------------------------------
          // Reconstruct E matrix
          // -----------------------------------------------------------------
          specfem::utilities::ComplexMatrix<4> e_mat;
          for (int k = 0; k < 16; ++k)
            e_mat.data[k] = d_e_mat(k);

          // -----------------------------------------------------------------
          // Per-frequency spectral accumulation
          // -----------------------------------------------------------------
          // Zero the field_f row for this point.
          for (int j = 0; j < 5; ++j)
            for (int ii = 0; ii < nf_k; ++ii)
              d_field_f(ipt, j, ii) = Cplx(0.0, 0.0);

          for (int ii = 0; ii < nf2_k; ++ii) {
            const double om_real = d_omvec(ii);
            const Cplx om(om_real, -sigma0_k);

            // stf_coeff (ref line 411-412)
            const Cplx stf_c =
                Cplx(d_stf_apod(ii), 0.0) *
                Kokkos::exp(Cplx(0.0, -1.0) * Cplx(om_real, 0.0) *
                            Cplx(Tdelay, 0.0));

            // Build bottom vector.
            Kokkos::Array<Cplx, 2> coeffs_arr;
            coeffs_arr[0] = d_coeffs(ii * 2 + 0);
            coeffs_arr[1] = d_coeffs(ii * 2 + 1);

            const specfem::utilities::ComplexVector<4> bot_vec =
                specfem::injection::fk_impl::build_bottom_vector(
                    wave_type, c1_k, c3_k, coeffs_arr);

            specfem::injection::fk_impl::PointSpectrum spec;

            const int base_el = ii * n_elastic_k * 16;

            if (is_elastic) {
              // chain_above = el_chain[ii, ilayer_el+1]  (identity when in hs
              // or when ilayer_el is the topmost finite layer).
              specfem::utilities::ComplexMatrix<4> chain_above;
              specfem::injection::fk_impl::set_identity(chain_above);
              if (!in_halfspace && (ilayer_el + 1) < n_elastic_k) {
                for (int k = 0; k < 16; ++k)
                  chain_above.data[k] =
                      d_el_chain(base_el + (ilayer_el + 1) * 16 + k);
              }

              specfem::injection::fk::ElasticIsotropicLayer point_layer_el;
              point_layer_el.p_velocity = d_el_vp(ilayer_el);
              point_layer_el.s_velocity = d_el_vs(ilayer_el);
              point_layer_el.density = d_el_rho(ilayer_el);
              point_layer_el.thickness = d_el_H(ilayer_el);

              specfem::injection::fk::ElasticIsotropicLayer hs_layer;
              hs_layer.p_velocity = d_el_vp(n_elastic_k - 1);
              hs_layer.s_velocity = d_el_vs(n_elastic_k - 1);
              hs_layer.density = d_el_rho(n_elastic_k - 1);
              hs_layer.thickness = d_el_H(n_elastic_k - 1);

              type_real xi1_pt = 0;
              type_real xim_pt = 0;
              if (points.lame_ratio_xi1.extent(0) > 0)
                xi1_pt = points.lame_ratio_xi1(ipt);
              if (points.lame_ratio_xim.extent(0) > 0)
                xim_pt = points.lame_ratio_xim(ipt);

              // above_all_layers: the point is above all finite elastic layers.
              // In the reference: ilayer == nlayer (the last elastic index in
              // 1-based = n_elastic-1 in 0-based elastic block).
              const bool above_all_el =
                  (ilayer_el == n_elastic_k - 1) && !in_halfspace;

              spec = specfem::injection::fk_impl::recover_elastic_point(
                  e_mat, chain_above, point_layer_el, hs_layer, om,
                  static_cast<type_real>(ray_p_k), in_halfspace, above_all_el,
                  static_cast<type_real>(zz), static_cast<type_real>(height),
                  bot_vec, stf_c, xi1_pt, xim_pt, compute_traction_k);

            } else {
              // Acoustic point.
              // elastic_chain_at_interface = el_chain[ii, 0]
              // (the product of all finite elastic layers from below the fluid
              // to the solid half-space; used to propagate the bot_vec up to
              // the fluid/solid interface).
              specfem::utilities::ComplexMatrix<4> elastic_chain_at_iface;
              specfem::injection::fk_impl::set_identity(elastic_chain_at_iface);
              // has_elastic_below: there is at least one elastic layer
              // (the half-space always exists, so this is always true when
              // there are elastic layers below the fluid).
              const bool has_el_below = (n_elastic_k >= 1);
              if (has_el_below && n_elastic_k > 1) {
                for (int k = 0; k < 16; ++k)
                  elastic_chain_at_iface.data[k] =
                      d_el_chain(base_el + 0 * 16 + k);
              }
              // When n_elastic_k == 1 (only half-space under fluid), no finite
              // elastic layers, so el_chain[ii,0] is identity — already set.

              // acoustic_chain_above = ac_chain[ii, ilayer_ac]
              specfem::utilities::ComplexMatrix<4> acoustic_chain_above;
              specfem::injection::fk_impl::set_identity(acoustic_chain_above);
              if (n_fluid_k > 0) {
                const int base_ac = ii * n_fluid_k * 16;
                for (int k = 0; k < 16; ++k)
                  acoustic_chain_above.data[k] =
                      d_ac_chain(base_ac + ilayer_ac * 16 + k);
              }

              specfem::injection::fk::AcousticLayer point_layer_ac;
              point_layer_ac.p_velocity = d_ac_vp(ilayer_ac);
              point_layer_ac.density = d_ac_rho(ilayer_ac);
              point_layer_ac.thickness = d_ac_H(ilayer_ac);

              // top_fluid_thickness = H of the topmost fluid layer.
              const double top_fluid_H =
                  (n_fluid_k > 0) ? static_cast<double>(d_ac_H(0)) : 0.0;

              // apply_free_surface_bc: point is at the very top of the fluid
              // column (ref line 504: abs(height - H[0]) < eps * H[0]).
              const bool apply_fbc =
                  (n_fluid_k > 0) &&
                  Kokkos::fabs(height - top_fluid_H) < 1.0e-6 * top_fluid_H;

              spec = specfem::injection::fk_impl::recover_acoustic_point(
                  e_mat, elastic_chain_at_iface, has_el_below,
                  acoustic_chain_above, point_layer_ac, om,
                  static_cast<type_real>(ray_p_k),
                  static_cast<type_real>(height),
                  static_cast<type_real>(top_fluid_H), apply_fbc, bot_vec,
                  stf_c);
            }

            // The recovery returns DISPLACEMENT in components 0,1. Apply the
            // requested kinematic derivative as (i*om)^deriv_order in the
            // frequency domain: 0 = displacement (Method 2), 1 = velocity
            // (Method 1 Stacey / SPECFEM3D Veloc_FK, ref
            // couple_with_injection.f90 lines 1269-1270), 2 = acceleration
            // (Method 2). The stress/pressure components (2..4) carry their own
            // om factor and are stored as-is.
            Cplx deriv_factor(1.0, 0.0);
            if (deriv_order_k == 1) {
              deriv_factor = Cplx(0.0, om_real);
            } else if (deriv_order_k == 2) {
              deriv_factor = Cplx(-om_real * om_real, 0.0);
            }
            spec.values[0] = spec.values[0] * deriv_factor;
            spec.values[1] = spec.values[1] * deriv_factor;

            for (int j = 0; j < 5; ++j)
              d_field_f(ipt, j, ii) = spec.values[j];
          } // end per-frequency loop

          // -----------------------------------------------------------------
          // Hermitian pad (ref lines 524-529)
          // ii = 2..nf2-1 (1-based) → 0-based ii_0 = 1..nf2-2
          // negative-frequency index = nf - ii_0
          // -----------------------------------------------------------------
          for (int ii_0 = 1; ii_0 <= nf2_k - 2; ++ii_0) {
            for (int j = 0; j < 5; ++j)
              d_field_f(ipt, j, nf_k - ii_0) =
                  Kokkos::conj(d_field_f(ipt, j, ii_0));
          }

          // -----------------------------------------------------------------
          // Inverse FFT for each of the 5 field components (ref lines 533-535)
          // -----------------------------------------------------------------
          for (int j = 0; j < 5; ++j) {
            specfem::utilities::fft::inverse(
                npow_k, &d_field_f(ipt, j, 0), -1.0, dt_sem_k,
                &d_field(ipt, j, 0), bit_reversal_ptr, twiddles_ptr);
          }

          // -----------------------------------------------------------------
          // t0 wrap (ref lines 537-551, nn > 0 branch).
          // Three-reversal left-rotation by nn positions (in-place, no
          // scratch).
          // -----------------------------------------------------------------
          if (nn_k != 0) {
            const int rot =
                (nn_k > 0) ? nn_k
                           : npts2_k + nn_k; // effective left rotation for nn<0

            if (rot > 0 && rot < npts2_k) {
              for (int j = 0; j < 5; ++j) {
                double *fcol = &d_field(ipt, j, 0);
                const int n = npts2_k;
                // Reverse full array
                for (int lo = 0, hi = n - 1; lo < hi; ++lo, --hi) {
                  const double tmp = fcol[lo];
                  fcol[lo] = fcol[hi];
                  fcol[hi] = tmp;
                }
                // Reverse first (n - rot) elements
                for (int lo = 0, hi = n - rot - 1; lo < hi; ++lo, --hi) {
                  const double tmp = fcol[lo];
                  fcol[lo] = fcol[hi];
                  fcol[hi] = tmp;
                }
                // Reverse last rot elements
                for (int lo = n - rot, hi = n - 1; lo < hi; ++lo, --hi) {
                  const double tmp = fcol[lo];
                  fcol[lo] = fcol[hi];
                  fcol[hi] = tmp;
                }
              }
            }
          }

          // -----------------------------------------------------------------
          // Exponential taper (ref lines 555-559)
          // field[i + j*npts2] *= exp(sigma0 * (i*dt_fk - t0 - Tdelay))
          // -----------------------------------------------------------------
          for (int i = 0; i < npts2_k; ++i) {
            const double taper_val = Kokkos::exp(
                sigma0_k * (static_cast<double>(i) * dt_fk_k - t0_k - Tdelay));
            for (int j = 0; j < 5; ++j)
              d_field(ipt, j, i) *= taper_val;
          }

          // -----------------------------------------------------------------
          // B-spline storage (ref lines 561-620)
          // Use device-resident scratch views d_col_capture / d_cvals_capture
          // (row ipt) to avoid stack-allocated arrays in GPU kernels.
          // -----------------------------------------------------------------
          double *col = &d_col_capture(ipt, 0);
          double *cvals = &d_cvals_capture(ipt, 0);

          // displacement x = field[0] * cos(phi)
          for (int i = 0; i < npts2_k; ++i)
            col[i] = d_field(ipt, 0, i) * Kokkos::cos(phi_k);
          specfem::utilities::cubic_bspline::compute_coefficients(col, npts2_k,
                                                                  cvals);
          for (int k = 0; k < nf_stored_k; ++k)
            d_displacement(ipt, 0, k) = static_cast<type_real>(cvals[k]);

          // displacement y = field[0] * sin(phi)
          for (int i = 0; i < npts2_k; ++i)
            col[i] = d_field(ipt, 0, i) * Kokkos::sin(phi_k);
          specfem::utilities::cubic_bspline::compute_coefficients(col, npts2_k,
                                                                  cvals);
          for (int k = 0; k < nf_stored_k; ++k)
            d_displacement(ipt, 1, k) = static_cast<type_real>(cvals[k]);

          // displacement z = field[1]
          for (int i = 0; i < npts2_k; ++i)
            col[i] = d_field(ipt, 1, i);
          specfem::utilities::cubic_bspline::compute_coefficients(col, npts2_k,
                                                                  cvals);
          for (int k = 0; k < nf_stored_k; ++k)
            d_displacement(ipt, 2, k) = static_cast<type_real>(cvals[k]);

          // pressure (acoustic points)
          if (!is_elastic && d_pressure.extent(0) > 0) {
            for (int i = 0; i < npts2_k; ++i)
              col[i] = d_field(ipt, 2, i);
            specfem::utilities::cubic_bspline::compute_coefficients(
                col, npts2_k, cvals);
            for (int k = 0; k < nf_stored_k; ++k)
              d_pressure(ipt, k) = static_cast<type_real>(cvals[k]);
          }

          // traction (elastic points with normals and Lamé factors)
          if (compute_traction_k && is_elastic && d_traction.extent(0) > 0 &&
              points.lame_ratio_bulk.extent(0) > 0) {

            const double cp = Kokkos::cos(phi_k);
            const double sp = Kokkos::sin(phi_k);
            const double bulk_ratio =
                static_cast<double>(points.lame_ratio_bulk(ipt));
            const double nx = (points.normal_x.extent(0) > 0)
                                  ? static_cast<double>(points.normal_x(ipt))
                                  : 0.0;
            const double ny = (points.normal_y.extent(0) > 0)
                                  ? static_cast<double>(points.normal_y(ipt))
                                  : 0.0;
            const double nz_val =
                (points.normal_z.extent(0) > 0)
                    ? static_cast<double>(points.normal_z(ipt))
                    : 1.0;

            // Traction rotation (ref 585-606), then spline per component.
            // Reuse 'col' (already declared above); process one component at a
            // time.
            for (int comp = 0; comp < 3; ++comp) {
              for (int lpts = 0; lpts < nf_stored_k; ++lpts) {
                const double sigma_rr = d_field(ipt, 2, lpts);
                const double sigma_rz = d_field(ipt, 3, lpts);
                const double sigma_zz = d_field(ipt, 4, lpts);
                const double sigma_tt = bulk_ratio * (sigma_rr + sigma_zz);

                const double Txx = sigma_rr * cp * cp + sigma_tt * sp * sp;
                const double Txy = cp * sp * (sigma_rr - sigma_tt);
                const double Txz = sigma_rz * cp;
                const double Tyy = sigma_rr * sp * sp + sigma_tt * cp * cp;
                const double Tyz = sigma_rz * sp;
                const double Tzz = sigma_zz;

                if (comp == 0)
                  col[lpts] = Txx * nx + Txy * ny + Txz * nz_val;
                else if (comp == 1)
                  col[lpts] = Txy * nx + Tyy * ny + Tyz * nz_val;
                else
                  col[lpts] = Txz * nx + Tyz * ny + Tzz * nz_val;
              }
              specfem::utilities::cubic_bspline::compute_coefficients(
                  col, nf_stored_k, cvals);
              for (int k = 0; k < nf_stored_k; ++k)
                d_traction(ipt, comp, k) = static_cast<type_real>(cvals[k]);
            }
          }
        }); // end Kernel 2

    Kokkos::fence();
  }

  // =========================================================================
  // Step 8: cosine taper (ref lines 633-648)
  // =========================================================================
  constexpr int taper_nlength = 20;
  if (nf_stored > 2 * taper_nlength) {
    auto d_displacement = result.displacement();
    auto d_traction = result.traction();
    auto d_pressure = result.pressure();
    const bool has_traction_k = result.has_traction();
    const bool has_pressure_k = result.has_pressure();

    Kokkos::parallel_for(
        "fk_cosine_taper", Kokkos::RangePolicy<>(0, npoints),
        KOKKOS_LAMBDA(int ipt) {
          for (int i = 1; i <= taper_nlength; ++i) {
            const double tv =
                0.5 * (1.0 - Kokkos::cos(specfem::injection::fk_impl::pi *
                                         static_cast<double>(i - 1) /
                                         static_cast<double>(taper_nlength)));
            for (int comp = 0; comp < 3; ++comp)
              d_displacement(ipt, comp, i - 1) = static_cast<type_real>(
                  tv * static_cast<double>(d_displacement(ipt, comp, i - 1)));

            if (has_pressure_k)
              d_pressure(ipt, i - 1) = static_cast<type_real>(
                  tv * static_cast<double>(d_pressure(ipt, i - 1)));

            if (has_traction_k)
              for (int comp = 0; comp < 3; ++comp)
                d_traction(ipt, comp, i - 1) = static_cast<type_real>(
                    tv * static_cast<double>(d_traction(ipt, comp, i - 1)));
          }
        });

    Kokkos::fence();
  }

  // =========================================================================
  // Step 9: sampling metadata
  // =========================================================================
  const double dt_min_fk = static_cast<double>(sizes.resampling_rate) *
                           static_cast<double>(window.dt);
  result.set_sampling(sizes.resampling_rate, static_cast<type_real>(dt_min_fk),
                      wave.origin_time);

  return result;
}
