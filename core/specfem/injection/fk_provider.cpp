#include "specfem/injection/fk_provider.hpp"

#include "specfem/injection/fk/boundary_coupling.hpp"
#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/solver.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag>
specfem::injection::fk_provider<DimensionTag>::fk_provider(
    specfem::injection::fk::LayeredModel model,
    specfem::injection::fk::IncidentWave wave,
    specfem::injection::fk::TimeWindow window,
    specfem::injection::injection_points points, bool compute_traction,
    specfem::injection::fk::field_derivative derivative, int window_steps,
    double consistency_tolerance)
    : model_(std::move(model)), wave_(std::move(wave)),
      window_(std::move(window)), points_(std::move(points)),
      compute_traction_(compute_traction), derivative_(derivative),
      window_steps_(window_steps),
      consistency_tolerance_(consistency_tolerance) {}

// ---------------------------------------------------------------------------
// initialize(): run FK solve and allocate the frame buffer
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag>
void specfem::injection::fk_provider<DimensionTag>::initialize() {
  // Derive the FK evaluation points (Lamé ratios from the local material) and
  // assert the mesh material matches the 1-D model at every point.
  const specfem::injection::fk::EvalPoints eval_points =
      specfem::injection::fk::eval_points_from_points(
          points_, model_, compute_traction_, consistency_tolerance_);

  result_ = specfem::injection::fk::solve(model_, wave_, window_, eval_points,
                                          compute_traction_, derivative_);

  const int nstep = window_.nstep;
  const int ncomp = compute_traction_ ? 6 : 3;
  const int npoints = points_.size();

  // Window length: honor an explicit request, else fit within a fixed memory
  // budget (a fixed 64-step window would be ~1.5 GB at 1e6 points).
  int window_steps;
  if (window_steps_ > 0) {
    window_steps = std::min(nstep, window_steps_);
  } else {
    constexpr std::size_t budget_bytes = std::size_t(256) << 20; // 256 MiB
    const std::size_t per_step =
        static_cast<std::size_t>(npoints) * ncomp * sizeof(type_real);
    const int fit = (per_step > 0) ? static_cast<int>(std::max<std::size_t>(
                                         1, budget_bytes / per_step))
                                   : nstep;
    window_steps = std::min(nstep, fit);
  }

  buffer_ = InjectionFrameBuffer(npoints, ncomp, window_steps);
  window_loaded_ = false;
}

// ---------------------------------------------------------------------------
// ensure_window(): reconstruct the full time series from B-spline coefficients
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag>
void specfem::injection::fk_provider<DimensionTag>::ensure_window(int istep) {
  if (window_loaded_ && buffer_.contains(istep)) {
    return;
  }
  const int nstep = window_.nstep;
  const int window_steps = buffer_.number_of_steps();
  const int start = (istep / window_steps) * window_steps;
  const int count = std::min(window_steps, nstep - start);
  result_.sample(start, count, buffer_.values());
  buffer_.set_window_start(start);
  window_loaded_ = true;
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
