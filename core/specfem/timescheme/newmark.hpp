#pragma once

#include "specfem/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/timescheme/timescheme.hpp"
#include "specfem_setup.hpp"

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
 * Specialized for forward and combined (adjoint) simulations.
 *
 * @tparam AssemblyFields Field assembly type containing wavefield data
 * @tparam SimulationType Either forward or combined simulation type
 */
template <typename AssemblyFields, specfem::simulation::type SimulationType>
class newmark;

/**
 * @brief Newmark scheme for forward simulation
 *
 * Forward-only time integration where the adjoint wavefield methods are
 * no-ops (return 0).
 *
 * @code
 * // Typical usage in a time loop
 * for (const auto [istep, dt] : scheme.iterate_forward()) {
 *   scheme.apply_predictor_phase_forward(medium_tag);
 *   // ... compute forces/accelerations ...
 *   scheme.apply_corrector_phase_forward(medium_tag);
 * }
 * @endcode
 */
template <typename AssemblyFields>
class newmark<AssemblyFields, specfem::simulation::type::forward>
    : public time_scheme {

public:
  constexpr static auto dimension_tag =
      AssemblyFields::dimension_tag; ///< Dimension tag

  constexpr static auto simulation_type =
      specfem::simulation::type::forward; ///< Wavefield tag

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
  newmark(AssemblyFields &fields, int nstep, const int nstep_between_samples,
          const type_real dt, const type_real t0)
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
   * Calls newmark_impl::predictor_phase_impl() on the forward wavefield with
   * positive timestep.
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
   * Calls newmark_impl::corrector_phase_impl() on the forward wavefield with
   * positive \f$\Delta t/2\f$.
   *
   * @param tag Medium tag for elements to apply the corrector phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_corrector_phase_forward(
      const specfem::element::medium_tag tag) override;

  /**
   * @brief Apply the predictor phase for backward simulation on fields within
   * the elements within a medium. (Empty implementation)
   *
   * @param tag Medium tag for elements to apply the predictor phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_predictor_phase_backward(
      const specfem::element::medium_tag tag) override {
    return 0;
  };

  /**
   * @brief  Apply the corrector phase for backward simulation on fields within
   * the elements within a medium. (Empty implementation)
   *
   * @param tag Medium tag for elements to apply the corrector phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_corrector_phase_backward(
      const specfem::element::medium_tag tag) override {
    return 0;
  };

public:
  /**
   * @brief Get the timescheme type
   *
   * @return specfem::enums::time_scheme::type Timescheme type
   */
  specfem::enums::time_scheme::type timescheme() const override {
    return specfem::enums::time_scheme::type::newmark;
  }

  /**
   * @brief Get the time increment
   *
   * @return type_real Time increment
   */
  type_real get_timestep() const override { return this->deltat; }

protected:
  type_real t0;               ///< Initial time
  type_real deltat;           ///< Time increment
  type_real deltatover2;      ///< Half time increment
  type_real deltasquareover2; ///< Half of squared time increment
  AssemblyFields fields;      ///< Assembly fields
};

/**
 * @brief Newmark scheme for combined forward-adjoint simulation
 *
 * Manages three wavefields: forward (adjoint source), adjoint, and backward.
 * Forward and adjoint integrate forward in time; backward integrates backward.
 * Used for computing sensitivity kernels and adjoint gradients.
 *
 * @code
 * // Forward phase
 * for (const auto [istep, dt] : scheme.iterate_forward()) {
 *   scheme.apply_predictor_phase_forward(medium_tag);
 *   // ... compute forces ...
 *   scheme.apply_corrector_phase_forward(medium_tag);
 * }
 * // Backward phase
 * for (const auto [istep, dt] : scheme.iterate_backward()) {
 *   scheme.apply_predictor_phase_backward(medium_tag);
 *   // ... compute forces ...
 *   scheme.apply_corrector_phase_backward(medium_tag);
 * }
 * @endcode
 */
template <typename AssemblyFields>
class newmark<AssemblyFields, specfem::simulation::type::combined>
    : public time_scheme {

public:
  constexpr static auto dimension_tag =
      AssemblyFields::dimension_tag; ///< Dimension tag

  constexpr static auto simulation_type =
      specfem::simulation::type::combined; ///< Wavefield tag
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
   * Calls newmark_impl::predictor_phase_impl() on the adjoint wavefield with
   * positive timestep.
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
   * Calls newmark_impl::corrector_phase_impl() on the adjoint wavefield with
   * positive \f$\Delta t/2\f$.
   *
   * @param tag Medium tag for elements to apply the corrector phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_corrector_phase_forward(
      const specfem::element::medium_tag tag) override;

  /**
   * @brief Apply the predictor phase for backward simulation on fields within
   * the elements within a medium.
   *
   * Calls newmark_impl::predictor_phase_impl() on the backward wavefield with
   * negative timestep (\f$-\Delta t\f$).
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
   * Calls newmark_impl::corrector_phase_impl() on the backward wavefield with
   * negative \f$\Delta t/2\f$.
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
   * @return specfem::enums::time_scheme::type Timescheme type
   */
  specfem::enums::time_scheme::type timescheme() const override {
    return specfem::enums::time_scheme::type::newmark;
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
