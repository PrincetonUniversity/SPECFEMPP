#ifndef _NEWMARK_HPP
#define _NEWMARK_HPP

#include "domain/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "specfem_setup.hpp"
#include "timescheme.hpp"
#include <ostream>

namespace specfem {
namespace TimeScheme {
/**
 * @brief Newmark timescheme
 *
 */
class Newmark : public specfem::TimeScheme::TimeScheme {

public:
  /**
   * @brief Get the timescheme type
   *
   * @return constexpr specfem::enums::time_scheme
   */
  specfem::enums::time_scheme::type timescheme() const override {
    return specfem::enums::time_scheme::type::newmark;
  }
  /**
   * @brief Construct a new Newmark timescheme object
   *
   * @param nstep maximum number of timesteps in the simulation
   * @param t0 Simulation start time
   * @param dt delta for the newmark timescheme
   */
  Newmark(const int nstep, const type_real t0, const type_real dt,
          const int nstep_between_samples);
  /**
   * @brief Return the status of simulation
   *
   * @return false if current step >= number of steps
   * @return true if current step < number of steps
   */
  bool status() const override { return (this->istep < this->nstep); }
  /**
   * @brief increment by one timestep, also updates the simulation time by dt
   *
   */
  void increment_time() override;
  /**
   * @brief Get the current simulation time
   *
   * @return type_real current time
   */
  type_real get_time() const override { return this->current_time; }
  /**
   * @brief Get the current timestep
   *
   * @return int current timestep
   */
  int get_timestep() const override { return this->istep; }
  /**
   * @brief reset current time to t0 and timestep to 0
   *
   */
  void reset_time() override;
  // void update_fields(specfem::Domain::Domain *domain_class){};
  /**
   * @brief Get the max timestep (nstep) of the simuation
   *
   * @return int max timestep
   */
  int get_max_timestep() override { return this->nstep; }
  /**
   * @brief Apply predictor phase of the timescheme
   *
   * @param domain_class Pointer to domain class to apply predictor phase
   */
  void apply_predictor_phase(
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
          field_dot_dot) override;
  /**
   * @brief Apply corrector phase of the timescheme
   *
   * @param domain_class Pointer to domain class to apply corrector phase
   */
  void apply_corrector_phase(
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
          field_dot_dot) override;
  /**
   * @brief Compute if seismogram needs to be calculated at this timestep
   *
   */
  bool compute_seismogram() const override {
    return (this->istep % nstep_between_samples == 0);
  };
  /**
   * @brief Get the current seismogram step
   *
   * @return int value of the current seismogram step
   */
  int get_seismogram_step() const override { return isig_step; }
  /**
   * @brief Get the max seismogram step
   *
   * @return int maximum value of seismogram step
   */
  virtual int get_max_seismogram_step() const override {
    return nstep / nstep_between_samples;
  }
  /**
   * @brief Increment seismogram step
   *
   */
  void increment_seismogram_step() override { isig_step++; }

  /**
   * @brief Get time increment
   *
   */
  type_real get_time_increment() const override { return this->deltat; }

  /**
   * @brief Log newmark timescheme information to console
   *
   */
  void print(std::ostream &out) const override;

private:
  type_real current_time;      ///< Current simulation time in seconds
  int istep = 0;               ///< Current simulation step
  type_real deltat;            ///< time increment (\f$ \delta t \f$)
  type_real deltatover2;       ///< \f$ \delta t / 2 \f$
  type_real deltatsquareover2; ///< \f$ \delta t^2 / 2 \f$
  int nstep;                   ///< Maximum value of timestep
  type_real t0;                ///< Simultion start time in seconds
  int nstep_between_samples;   ///< Number of time steps between seismogram
                               ///< outputs
  int isig_step = 0;           ///< current seismogram step
};
} // namespace TimeScheme
} // namespace specfem
#endif
