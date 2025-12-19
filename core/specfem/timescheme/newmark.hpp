#pragma once

#include "enumerations/simulation.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/timescheme.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace time_scheme {

/**
 * @brief Newmark Time Scheme
 *
 */
template <typename AssemblyFields, specfem::simulation::type SimulationType>
class newmark;

/**
 * @brief Template specialization for the forward simulation
 *
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
   * @name Convert timescheme to string
   */
  std::string to_string() const override;

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
 * @brief Template specialization for the adjoint simulation
 *
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
   * @name Convert timescheme to string
   */
  std::string to_string() const override;

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
  type_real t0;     ///< Initial time
  type_real deltat; ///< Time increment
  type_real deltatover2;
  type_real deltasquareover2;
  AssemblyFields fields; ///< Assembly fields
};

} // namespace time_scheme
} // namespace specfem
