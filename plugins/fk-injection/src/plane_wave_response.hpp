#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/reference_convention.hpp"
#include "fk_injection/time_window.hpp"
#include "fk_injection/utilities/complex_matrix.hpp"
#include "impl/layer_stack.hpp"
#include <complex>
#include <vector>

namespace fk_injection::impl {

/**
 * @brief Frequency-domain quantities shared by every series row.
 *
 * The time-placement bookkeeping (t0 shift, delay pad) is applied where the
 * series are synthesized and sampled.
 */
struct FkSetup {
  fk_injection::FkSizes sizes;      ///< Working-array sizes.
  double p = 0.0;                   ///< Horizontal ray parameter (s/m).
  double phi = 0.0;                 ///< Azimuth (rad).
  double dt_min = 0.0;              ///< Resampled time step N_P * dt (s).
  std::complex<double> amplitude{}; ///< Incident mode amplitude in the
                                    ///< half-space.
  fk_injection::incident_wave_type wave_type =
      fk_injection::incident_wave_type::p; ///< Incident wave type.
  fk_injection::ComplexMatrix<4> e_mat;    ///< Half-space eigenmatrix E.
  std::vector<double> omega;       ///< Angular frequencies, size nf_stored+1.
  std::vector<double> apodization; ///< Gaussian STF spectrum, size nf_stored+1.
  int pad_samples =
      0; ///< ceil(max delay / dt_min); keeps sample abscissae >= 0.
  int shift_samples = 0; ///< t0 series rotation nn - pad (left rotation), nn =
                         ///< origin_time / dt_min, floored or truncated.
  double abscissa_pad = 0.0;  ///< pad_samples plus the part of origin_time /
                              ///< dt_min the rotation can't express, in [0, 1).
  double delay_stretch = 1.0; ///< Scale on every point's delay: 1, or N/(N-1)
                              ///< to reproduce SPECFEM3D's frequency grid.
  double normalization_dt = 0.0; ///< Step the inverse FFT divides by: dt_min,
                                 ///< or the SEM step to reproduce SPECFEM3D.
};

/**
 * @brief Time for the fluid block's reverberation to decay to 1e-4.
 *
 * A wave trapped between the free surface and the fluid's base loses a factor
 * \f$|R|\f$ per round trip \f$\tau = 2\sum_i H_i q_i\f$, where \f$R\f$ is the
 * fluid-to-solid P reflection coefficient at ray parameter \f$p\f$. The decay
 * time is \f$\tau \ln(10^{-4}) / \ln|R|\f$. An estimate: reflections between
 * fluid layers and reverberations in solid layers are ignored.
 *
 * @param model Layered velocity model.
 * @param p     Horizontal ray parameter (s/m).
 * @return Decay time in s; 0 without a fluid or if the wave is evanescent in
 *         it.
 * @throws std::runtime_error if the fluid's base reflects totally.
 */
double fluid_reverberation_time(const fk_injection::LayeredModel &model,
                                double p);

/**
 * @brief Reject a ray parameter outside the range FK supports.
 *
 * Every vertical slowness has to be real and non-zero, so @p p must be positive
 * and below @f$ 1/v @f$ for every layer velocity @f$ v @f$: @f$ p = 1/v @f$ is
 * 0/0 in the propagators. Since @f$ v_P @f$ of the half-space is among those
 * velocities, incident SV is limited to take-off angles below
 * @f$ \arcsin(v_S/v_P) @f$.
 *
 * @param model Layered velocity model.
 * @param p     Horizontal ray parameter (s/m).
 * @throws specfem::plugin::Unsupported if @p p is out of range.
 */
void check_supported_slowness(const fk_injection::LayeredModel &model,
                              double p);

} // namespace fk_injection::impl

namespace fk_injection {

/** @brief Horizontal extent of the injection points of every rank, in m. */
struct HorizontalBox {
  double x_min = 0;
  double x_max = 0;
  double y_min = 0;
  double y_max = 0;
};

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
   * The window covers the time-window length plus the delay span over the box,
   * and in the exact convention at least the span plus
   * @c impl::fluid_reverberation_time, so a fluid's reverberation decays
   * within one FFT period instead of wrapping onto the run.
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
  static impl::FkSetup prepare(const LayeredModel &model,
                               const IncidentWave &wave,
                               const TimeWindow &window, double p,
                               double time_delay_min, double time_delay_max,
                               ReferenceConvention convention);

  LayeredModel model_;
  double ray_parameter_ = 0;
  double azimuth_ = 0;
  double origin_x_ = 0;
  double origin_y_ = 0;
  double vertical_delay_ = 0; ///< @f$ \eta_p(z_{ref} - z_0) @f$
  int number_of_steps_ = 0;
  impl::FkSetup setup_{};
  impl::InterfaceStates states_{};
};

} // namespace fk_injection
