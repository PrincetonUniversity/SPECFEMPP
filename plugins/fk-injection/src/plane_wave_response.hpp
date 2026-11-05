#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/reference_convention.hpp"
#include "fk_injection/time_window.hpp"
#include "impl/fk_setup.hpp"
#include "impl/layer_stack.hpp"
#include <vector>

namespace fk_injection {

/** @brief Horizontal extent of the injection points of every rank, in m. */
struct HorizontalBox {
  double x_min = 0;
  double x_max = 0;
  double y_min = 0;
  double y_max = 0;
};

/**
 * @brief Smallest box holding the points (x[i], y[i]).
 *
 * For callers that see every point, such as tests; ranks of a partitioned mesh
 * must share the mesh box instead.
 *
 * @param x x-coordinates in m
 * @param y y-coordinates in m, as many as @p x
 * @return The box; all zero without points
 */
HorizontalBox bounding_box(const std::vector<type_real> &x,
                           const std::vector<type_real> &y);

/**
 * @brief The layered model's response to an incident plane wave, at every
 *        stored frequency.
 *
 * Holds no points: the window is sized from the delays at the box corners,
 * which bound the delay of every point inside it, so every rank sharing the
 * box builds the same response.
 */
class PlaneWaveResponse {
public:
  /**
   * @brief Size the window, check it, and solve the layer stack.
   *
   * @param model      Layered velocity model
   * @param wave       Incident plane wave
   * @param window     Simulation time step and windowing parameters
   * @param box        Horizontal extent of the points
   * @param convention See @c ReferenceConvention
   * @throws specfem::plugin::Unsupported if the ray parameter is out of range
   *         (see @c impl::check_supported_slowness)
   * @throws std::runtime_error if the window doesn't cover every step at the
   *         box corners, or the run needs FK times outside one FFT period
   */
  PlaneWaveResponse(
      const LayeredModel &model, const IncidentWave &wave,
      const TimeWindow &window, const HorizontalBox &box,
      ReferenceConvention convention = ReferenceConvention::exact);

  /**
   * @brief Plane-wave delay of the horizontal position (x, y).
   *
   * @f$ \tau = p[(x-x_0)\cos\phi + (y-y_0)\sin\phi] + \eta_p(z_{ref} - z_0)
   * @f$, with @f$ z_{ref} @f$ the half-space top and @f$ \eta_p @f$ the
   * half-space P slowness for both wave types: the FK time zero is where the
   * incident wave crosses the half-space top, as SPECFEM3D's Z_REF_for_FK.
   *
   * @param x x-coordinate in m
   * @param y y-coordinate in m
   * @return Delay in s
   */
  double time_delay(double x, double y) const;

  /**
   * @brief Spline abscissa of step 0 at (x, y), in resampled samples.
   * @param x x-coordinate in m
   * @param y y-coordinate in m
   * @return Pad minus the stretched delay over @c dt_min
   */
  double abscissa_offset(double x, double y) const;

  /**
   * @brief Check that every step at (x, y) maps inside the stored series.
   *
   * Spline evaluation clamps out-of-range abscissae into a silent DC offset.
   * The check uses the unstretched delay: the SPECFEM3D convention may ride
   * the window edge by up to the N/(N-1) stretch and rely on the same clamp
   * the reference does.
   *
   * @param x x-coordinate in m
   * @param y y-coordinate in m
   * @throws std::runtime_error if a step falls outside
   */
  void check_coverage(double x, double y) const;

  const LayeredModel &model() const { return model_; }
  const impl::FkSetup &setup() const { return setup_; }
  const impl::InterfaceStates &states() const { return states_; }

private:
  LayeredModel model_;
  double origin_x_ = 0;
  double origin_y_ = 0;
  double vertical_delay_ = 0; ///< @f$ \eta_p(z_{ref} - z_0) @f$
  int number_of_steps_ = 0;
  impl::FkSetup setup_{};
  impl::InterfaceStates states_{};
};

} // namespace fk_injection
