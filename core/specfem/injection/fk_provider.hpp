#pragma once

/**
 * @brief FK-method concrete injection provider.
 *
 * Implements @c injection_provider by running the wavenumber-frequency (FK)
 * plane-wave solver and reconstructing the full time series into a uniform
 * @c InjectionFrameBuffer via cubic-B-spline interpolation.
 */

#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/fk_result.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/time_window.hpp"
#include "specfem/injection/injection_frame_buffer.hpp"
#include "specfem/injection/injection_provider.hpp"

namespace specfem {
namespace injection {

/**
 * @brief Concrete injection provider backed by the FK plane-wave solver.
 *
 * @tparam DimensionTag Spatial dimension (@c
 * specfem::element::dimension_tag::dim3 is the supported instantiation; dim2 is
 * available but untested against a 2-D mesh).
 *
 * Workflow:
 * -# Construct with the four FK input structs.
 * -# Call @c initialize() once — runs the FK solve and stores B-spline
 *    coefficient Views on the device.
 * -# Call @c ensure_window(istep) before the time loop (or at least once
 *    before the first kernel launch) — reconstructs the full time series into
 *    the @c InjectionFrameBuffer.
 * -# The stiffness kernel reads values via @c frame_buffer().load_on_device.
 *
 * Component layout in the frame buffer:
 * - components 0–2: displacement (x, y, z)
 * - components 3–5: traction (x, y, z) — only when @p compute_traction is true
 */
template <specfem::element::dimension_tag DimensionTag>
class fk_provider : public injection_provider<DimensionTag> {
public:
  /**
   * @brief Construct an FK provider with the given solver inputs.
   *
   * @param model            1-D horizontally-layered velocity model.
   * @param wave             Incident plane-wave descriptor.
   * @param window           Simulation time-step and windowing parameters.
   * @param points           Device-resident evaluation-point coordinates and
   *                         optional normals / Lamé ratios.
   * @param compute_traction When true, compute and store traction components
   *                         in the frame buffer (components 3–5).  Requires
   *                         @p points to have non-zero normal and Lamé Views.
   */
  fk_provider(specfem::injection::fk::LayeredModel model,
              specfem::injection::fk::IncidentWave wave,
              specfem::injection::fk::TimeWindow window,
              specfem::injection::fk::EvalPoints points,
              bool compute_traction = true);

  /**
   * @brief Run the FK solve; allocates B-spline coefficient Views on device.
   *
   * Must be called once before @c ensure_window.
   */
  void initialize() override;

  /**
   * @brief Reconstruct the full time series into the frame buffer (first call
   * only; subsequent calls are no-ops).
   *
   * First-cut implementation: the entire series is reconstructed at once
   * regardless of @p istep.  Windowed streaming is deferred to a later
   * sub-plan.
   *
   * @param istep Simulation time step being requested (used for future
   *              windowed streaming; currently ignored after the first call).
   */
  void ensure_window(int istep) override;

  /**
   * @brief Return the populated device frame buffer.
   *
   * @return const reference to the InjectionFrameBuffer populated by
   *         @c ensure_window
   */
  const InjectionFrameBuffer &frame_buffer() const override;

private:
  specfem::injection::fk::LayeredModel model_; ///< Velocity model
  specfem::injection::fk::IncidentWave wave_;  ///< Incident wave descriptor
  specfem::injection::fk::TimeWindow window_;  ///< Time-window parameters
  specfem::injection::fk::EvalPoints points_;  ///< Evaluation points
  bool compute_traction_ = true;               ///< Whether to store traction
  specfem::injection::fk::FkResult result_{};  ///< B-spline coefficient result
  InjectionFrameBuffer buffer_{};              ///< Reconstructed time series
  bool reconstructed_ = false; ///< True after first ensure_window
};

} // namespace injection
} // namespace specfem
