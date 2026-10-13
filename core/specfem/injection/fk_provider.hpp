#pragma once

/**
 * @brief FK-method concrete injection provider.
 *
 * Implements @c injection_provider by running the wavenumber-frequency (FK)
 * plane-wave solver and reconstructing the full time series into a uniform
 * @c InjectionFrameBuffer via cubic-B-spline interpolation.
 */

#include "specfem/injection/fk/field_derivative.hpp"
#include "specfem/injection/fk/fk_result.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/time_window.hpp"
#include "specfem/injection/injection_frame_buffer.hpp"
#include "specfem/injection/injection_points.hpp"
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
 * -# Call @c ensure_window(istep) before each kernel launch — reconstructs the
 *    window of steps containing @p istep into the @c InjectionFrameBuffer.
 * -# The stiffness kernel reads values via @c frame_buffer().load_on_device.
 *
 * Component layout in the frame buffer:
 * - components 0–2: displacement (x, y, z)
 * - components 3–5: traction (x, y, z) — only when @p compute_traction is true
 * - component 6: pressure — only when @p compute_traction is true and the model
 *   has a fluid layer (acoustic faces; zero at elastic points)
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
   * @param points           Method-agnostic injection points (geometry, medium,
   *                         local material). The FK Lamé ratios are derived
   *                         internally from @c points.vp / @c points.vs in
   *                         @c initialize().
   * @param compute_traction When true, compute and store traction components
   *                         in the frame buffer (components 3–5). Requires
   *                         @p points to carry outward normals.
   * @param derivative       Kinematic quantity for the vector components
   *                         (displacement / velocity / acceleration); the
   *                         Stacey absorbing method requires velocity.
   * @param window_steps     Number of SEM steps held in the frame buffer at
   *                         once; 0 picks an automatic size within a memory
   *                         budget. Clamped to the total step count.
   * @param consistency_tolerance Relative tolerance for the model-vs-mesh
   *                         material check performed in @c initialize().
   */
  fk_provider(specfem::injection::fk::LayeredModel model,
              specfem::injection::fk::IncidentWave wave,
              specfem::injection::fk::TimeWindow window,
              specfem::injection::injection_points points,
              bool compute_traction = true,
              specfem::injection::fk::field_derivative derivative =
                  specfem::injection::fk::field_derivative::displacement,
              int window_steps = 0, double consistency_tolerance = 1e-3);

  /**
   * @brief Run the FK solve; allocates B-spline coefficient Views on device.
   *
   * Must be called once before @c ensure_window.
   */
  void initialize() override;

  /**
   * @brief Ensure the frame buffer holds the window containing @p istep.
   *
   * If @p istep already lies in the buffered window this is a no-op. Otherwise
   * the aligned window @c [start, start + W) with @c start = (istep / W) * W is
   * reconstructed via @c FkResult::sample. Aligned starts make the result
   * independent of call order. Callers must (re)capture the frame buffer after
   * this call, since the window contents change.
   *
   * @param istep Simulation time step being requested.
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
  specfem::injection::fk::LayeredModel model_;  ///< Velocity model
  specfem::injection::fk::IncidentWave wave_;   ///< Incident wave descriptor
  specfem::injection::fk::TimeWindow window_;   ///< Time-window parameters
  specfem::injection::injection_points points_; ///< Injection points (generic)
  bool compute_traction_ = true;                ///< Whether to store traction
  specfem::injection::fk::field_derivative derivative_ =
      specfem::injection::fk::field_derivative::displacement; ///< Vector
                                                              ///< quantity
  int window_steps_ = 0;                ///< Requested window size (0=auto)
  double consistency_tolerance_ = 1e-3; ///< Model-vs-mesh material tolerance
  specfem::injection::fk::FkResult result_{}; ///< B-spline coefficient result
  InjectionFrameBuffer buffer_{};             ///< Current reconstructed window
  bool window_loaded_ = false; ///< True once a window has been sampled
};

} // namespace injection
} // namespace specfem
