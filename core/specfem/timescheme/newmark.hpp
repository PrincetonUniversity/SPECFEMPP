#pragma once

#include "specfem/assembly/fields.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/timescheme/timescheme.hpp"

namespace specfem {
namespace time_scheme {

/**
 * @brief Newmark time scheme implementation
 *
 * Template implementations for predictor and corrector phases of the Newmark
 * time integration scheme. Uses Kokkos parallelism for GPU/CPU execution.
 */
namespace newmark_impl {
/**
 * @brief Implements Newmark **Corrector Phase**
 *
 * **Corrector Phase** updates velocity using the new acceleration computed
 * after the predictor:
 *
 * \f[ v^{n+1} = v^{n+\frac{1}{2}} + \frac{\Delta t}{2} a^{n+1} \f]
 *
 * @tparam DimensionTag 2D or 3D simulation
 * @tparam MediumTag Medium type (elastic, acoustic, etc.)
 * @tparam WavefieldType Forward, adjoint, or backward wavefield
 * @param field Simulation field containing velocity and acceleration
 * @param deltatover2 Half of the timestep (dt/2, or -dt/2 for backward)
 * @return Number of degrees of freedom updated
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::simulation::field_type WavefieldType>
int corrector_phase_impl(
    const specfem::assembly::simulation_field<DimensionTag, WavefieldType>
        &field,
    const type_real deltatover2);

/**
 * @brief Implements Newmark **Predictor Phase**
 *
 *
 * **Predictor Phase** updates displacement and velocity, then zeros
 * acceleration:
 *
 * \f[
 * \begin{aligned}
 *   u^{n+1} &= u^n + \Delta t \, v^n + \frac{\Delta t^2}{2} a^n \\
 *   v^{n+\frac{1}{2}} &= v^n + \frac{\Delta t}{2} a^n \\
 *   a^{n+1} &= 0
 * \end{aligned}
 * \f]
 *
 * @tparam DimensionTag 2D or 3D simulation
 * @tparam MediumTag Medium type (elastic, acoustic, etc.)
 * @tparam WavefieldType Forward, adjoint, or backward wavefield
 * @param field Simulation field containing displacement, velocity, acceleration
 * @param deltat Timestep (dt, or -dt for backward integration)
 * @param deltatover2 Half timestep (dt/2, or -dt/2 for backward)
 * @param deltasquareover2 Half of squared timestep (dt²/2)
 * @return Number of degrees of freedom updated
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::simulation::field_type WavefieldType>
int predictor_phase_impl(
    const specfem::assembly::simulation_field<DimensionTag, WavefieldType>
        &field,
    const type_real deltat, const type_real deltatover2,
    const type_real deltasquareover2);
} // namespace newmark_impl

/**
 * @brief Newmark time integration scheme implementation
 *
 * Implements the Newmark-beta method for time integration of the wave
 * equation. This second-order accurate scheme uses predictor-corrector
 * steps:
 *
 * Behavior is selected at compile time from the simulation type.
 *
 * @tparam AssemblyFields Field assembly type containing wavefield data
 * @tparam SimulationType Forward, combined, or combined_undoatt simulation type
 */
template <typename AssemblyFields, specfem::simulation::type SimulationType>
class newmark : public time_scheme {

public:
  constexpr static auto dimension_tag =
      AssemblyFields::dimension_tag; ///< Dimension tag

  constexpr static auto simulation_type = SimulationType; ///< Simulation tag

  /**
   * @name Constructors
   */
  ///@{

  /**
   * @brief Construct a newmark time scheme object
   *
   * @param nstep Maximum number of timesteps
   * @param nstep_between_samples Number of timesteps between output seismogram
   * samples
   * @param dt Time increment
   * @param t0 Initial time
   */
  newmark(AssemblyFields &fields, const int nstep,
          const int nstep_between_samples, const type_real dt,
          const type_real t0)
      : time_scheme(nstep, nstep_between_samples, dt), deltat(dt),
        deltatover2(dt / 2.0), deltasquareover2(dt * dt / 2.0), t0(t0),
        fields(fields) {}

  ///@}

  /**
   * @brief Convert time scheme to string representation
   *
   * @return String describing Newmark scheme configuration
   */
  std::string to_string() const override;

  /**
   * @brief Print time scheme details to output stream
   *
   * @param out Output stream
   */
  void print(std::ostream &out) const override;

  /**
   * @brief Apply the predictor phase for forward simulation on fields within
   * the elements within a medium.
   *
   * Advances the forward wavefield for forward and combined_undoatt
   * simulations. Other simulation types return 0.
   *
   * @param tag Medium tag for elements to apply the predictor phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_predictor_phase_forward(
      const specfem::element::medium_tag tag) override;

  /**
   * @brief Apply the corrector phase for forward simulation on fields within
   * the elements within a medium.
   *
   * Advances the forward wavefield for forward and combined_undoatt
   * simulations. Other simulation types return 0.
   *
   * @param tag Medium tag for elements to apply the corrector phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_corrector_phase_forward(
      const specfem::element::medium_tag tag) override;

  /**
   * @brief Apply the predictor phase for adjoint simulation on fields within
   * the elements within a medium.
   *
   * Advances the adjoint wavefield for combined and combined_undoatt
   * simulations. Other simulation types return 0.
   *
   * @param tag Medium tag for elements to apply the predictor phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_predictor_phase_adjoint(
      const specfem::element::medium_tag tag) override;

  /**
   * @brief Apply the corrector phase for adjoint simulation on fields within
   * the elements within a medium.
   *
   * Advances the adjoint wavefield for combined and combined_undoatt
   * simulations. Other simulation types return 0.
   *
   * @param tag Medium tag for elements to apply the corrector phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_corrector_phase_adjoint(
      const specfem::element::medium_tag tag) override;

  /**
   * @brief Apply the predictor phase for backward simulation on fields within
   * the elements within a medium.
   *
   * Advances the backward wavefield for combined simulations. Other simulation
   * types return 0.
   *
   * @param tag  Medium tag for elements to apply the predictor phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_predictor_phase_backward(
      const specfem::element::medium_tag tag) override;

  /**
   * @brief  Apply the corrector phase for backward simulation on fields within
   * the elements within a medium.
   *
   * Advances the backward wavefield for combined simulations. Other simulation
   * types return 0.
   *
   * @param tag  Medium tag for elements to apply the corrector phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_corrector_phase_backward(
      const specfem::element::medium_tag tag) override;

public:
  /**
   * @brief Get the timescheme type
   *
   * @return specfem::time_scheme::type Timescheme type
   */
  specfem::time_scheme::type timescheme() const override {
    return specfem::time_scheme::type::newmark;
  }

  /**
   * @brief Get the time increament
   *
   * @return type_real Time increment
   */
  type_real get_timestep() const override { return this->deltat; }

protected:
  type_real t0;               ///< Initial time
  type_real deltat;           ///< Time increment
  type_real deltatover2;      ///< Half time increment (dt/2)
  type_real deltasquareover2; ///< Half squared time increment (dt²/2)
  AssemblyFields fields;      ///< Assembly fields
};

} // namespace time_scheme
} // namespace specfem
