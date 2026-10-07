#include "specfem/injection/fk_provider.hpp"

#include "specfem/injection/fk/solver.hpp"
#include "specfem/utilities/cubic_bspline.hpp"
#include <Kokkos_Core.hpp>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag>
specfem::injection::fk_provider<DimensionTag>::fk_provider(
    specfem::injection::fk::LayeredModel model,
    specfem::injection::fk::IncidentWave wave,
    specfem::injection::fk::TimeWindow window,
    specfem::injection::fk::EvalPoints points, bool compute_traction)
    : model_(std::move(model)), wave_(std::move(wave)),
      window_(std::move(window)), points_(std::move(points)),
      compute_traction_(compute_traction) {}

// ---------------------------------------------------------------------------
// initialize(): run FK solve and allocate the frame buffer
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag>
void specfem::injection::fk_provider<DimensionTag>::initialize() {
  result_ = specfem::injection::fk::solve(model_, wave_, window_, points_,
                                          compute_traction_);

  const int nstep = window_.nstep;
  const int ncomp = compute_traction_ ? 6 : 3;
  const int npoints = points_.size();

  buffer_ = InjectionFrameBuffer(npoints, ncomp, nstep);
}

// ---------------------------------------------------------------------------
// ensure_window(): reconstruct the full time series from B-spline coefficients
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag>
void specfem::injection::fk_provider<DimensionTag>::ensure_window(
    int /*istep*/) {
  if (reconstructed_) {
    return;
  }

  const int nstep = window_.nstep;
  const int npoints = result_.number_of_points();
  const int ncoef = result_.coefficient_count();
  const int np_resamp = result_.resampling_rate();

  // Capture Views by value for the device lambda (Kokkos reference semantics).
  const auto disp_view = result_.displacement(); // [npoints][3][ncoef]
  const auto trac_view =
      result_.traction(); // [npoints][3][ncoef]; empty if no traction
  auto buf_values = buffer_.values(); // [npoints][ncomp][nstep]

  const bool do_traction = compute_traction_;

  // Per-thread double scratch for the coefficient row passed to evaluate().
  // We use a 1-D scratch View sized to ncoef, allocated once on host and
  // broadcast to threads via a TeamPolicy scratch level.  However, to avoid
  // TeamPolicy complexity on the first-cut, we allocate a 2-D host-side
  // scratch buffer (npoints × ncoef), deep-copy the float coefficients to
  // double, and run the reconstruction in a RangePolicy over
  // (npoints × ncomp × nstep).
  //
  // Float→double copy: the FkResult Views are type_real (float when default
  // precision).  evaluate() requires double*.  We build a full double mirror
  // of the coefficient arrays on the host, then upload to a device double View
  // for use inside the kernel.

  // Build host double mirrors of the coefficient arrays.
  auto h_disp = Kokkos::create_mirror_view(disp_view);
  Kokkos::deep_copy(h_disp, disp_view);

  // Allocate device double Views for the coefficient data.
  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> disp_d(
      "fk_disp_double", npoints, 3, ncoef);
  {
    auto h_disp_d = Kokkos::create_mirror_view(disp_d);
    for (int ip = 0; ip < npoints; ++ip)
      for (int ic = 0; ic < 3; ++ic)
        for (int k = 0; k < ncoef; ++k)
          h_disp_d(ip, ic, k) = static_cast<double>(h_disp(ip, ic, k));
    Kokkos::deep_copy(disp_d, h_disp_d);
  }

  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> trac_d(
      "fk_trac_double", 0, 0, 0);
  if (do_traction) {
    auto h_trac = Kokkos::create_mirror_view(trac_view);
    Kokkos::deep_copy(h_trac, trac_view);
    trac_d = Kokkos::View<double ***, Kokkos::DefaultExecutionSpace>(
        "fk_trac_double", npoints, 3, ncoef);
    auto h_trac_d = Kokkos::create_mirror_view(trac_d);
    for (int ip = 0; ip < npoints; ++ip)
      for (int ic = 0; ic < 3; ++ic)
        for (int k = 0; k < ncoef; ++k)
          h_trac_d(ip, ic, k) = static_cast<double>(h_trac(ip, ic, k));
    Kokkos::deep_copy(trac_d, h_trac_d);
  }

  // Reconstruct: for each (point, component, step), evaluate the B-spline at
  // abscissa = istep / np_resamp (maps SEM step to resampled-grid coordinate).
  const int ncomp = do_traction ? 6 : 3;
  const int total = npoints * ncomp * nstep;

  Kokkos::parallel_for(
      "fk_provider_reconstruct",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, total),
      KOKKOS_LAMBDA(int idx) {
        const int istep_idx = idx % nstep;
        const int tmp = idx / nstep;
        const int icomp = tmp % ncomp;
        const int ipoint = tmp / ncomp;

        const double abscissa =
            static_cast<double>(istep_idx) / static_cast<double>(np_resamp);

        double val = 0.0;
        if (icomp < 3) {
          // Displacement component icomp.
          val = specfem::utilities::cubic_bspline::evaluate(
              &disp_d(ipoint, icomp, 0), ncoef, abscissa);
        } else {
          // Traction component (icomp - 3).
          const int tcomp = icomp - 3;
          val = specfem::utilities::cubic_bspline::evaluate(
              &trac_d(ipoint, tcomp, 0), ncoef, abscissa);
        }

        buf_values(ipoint, icomp, istep_idx) = static_cast<type_real>(val);
      });

  Kokkos::fence();
  reconstructed_ = true;
}

// ---------------------------------------------------------------------------
// frame_buffer()
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag>
const specfem::injection::InjectionFrameBuffer &
specfem::injection::fk_provider<DimensionTag>::frame_buffer() const {
  return buffer_;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template class specfem::injection::fk_provider<
    specfem::element::dimension_tag::dim3>;

template class specfem::injection::fk_provider<
    specfem::element::dimension_tag::dim2>;
