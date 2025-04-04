#pragma once

#include "enumerations/simulation.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem_setup.hpp"
#include "timescheme.hpp"

namespace specfem {
namespace time_scheme {

/**
 * @brief Newmark Time Scheme
 *
 * @tparam Simulation Simulation type on which this time scheme is applied
 */
template <specfem::simulation::type Simulation> class newmark;

/**
 * @brief Template specialization for the forward simulation
 *
 */
template <>
class newmark<specfem::simulation::type::forward> : public time_scheme {

public:
  constexpr static auto simulation_type =
      specfem::wavefield::simulation_field::forward; ///< Wavefield tag

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
  newmark(const int nstep, const int nstep_between_samples, const type_real dt,
          const type_real t0)
      : time_scheme(nstep, nstep_between_samples, dt), deltat(dt),
        deltatover2(dt / 2.0), deltasquareover2(dt * dt / 2.0), t0(t0) {}

  ///@}

  /**
   * @name Print timescheme details
   */
  void print(std::ostream &out) const override;

  /**
   * @brief Apply the predictor phase for forward simulation on fields within
   * the elements within a medium.
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

  void link_assembly(const specfem::compute::assembly &assembly) override {
    field = assembly.fields.forward;
  }

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

private:
  type_real t0;     ///< Initial time
  type_real deltat; ///< Time increment
  type_real deltatover2;
  type_real deltasquareover2;
  specfem::compute::simulation_field<
      specfem::wavefield::simulation_field::forward>
      field; ///< forward wavefield
};

/**
 * @brief Template specialization for the adjoint simulation
 *
 */
template <>
class newmark<specfem::simulation::type::combined> : public time_scheme {

public:
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
  newmark(const int nstep, const int nstep_between_samples, const type_real dt,
          const type_real t0)
      : time_scheme(nstep, nstep_between_samples, dt), deltat(dt),
        deltatover2(dt / 2.0), deltasquareover2(dt * dt / 2.0), t0(t0) {}

  ///@}

  /**
   * @name Print timescheme details
   */
  void print(std::ostream &out) const override;

  /**
   * @brief Apply the predictor phase for forward simulation on fields within
   * the elements within a medium.
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
   * @param tag  Medium tag for elements to apply the corrector phase
   * @return int Returns the number of degrees of freedom updated within the
   * medium
   */
  int apply_corrector_phase_backward(
      const specfem::element::medium_tag tag) override;

  void link_assembly(const specfem::compute::assembly &assembly) override {
    adjoint_field = assembly.fields.adjoint;
    backward_field = assembly.fields.backward;
  }

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

private:
  type_real t0;     ///< Initial time
  type_real deltat; ///< Time increment
  type_real deltatover2;
  type_real deltasquareover2;
  specfem::compute::simulation_field<
      specfem::wavefield::simulation_field::adjoint>
      adjoint_field; ///< adjoint wavefield
  specfem::compute::simulation_field<
      specfem::wavefield::simulation_field::backward>
      backward_field; ///< backward wavefield
};

} // namespace time_scheme
} // namespace specfem
