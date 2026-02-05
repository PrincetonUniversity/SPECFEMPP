#pragma once
#include "enumerations/interface.hpp"
#include "specfem/timescheme/impl/time_iterators.hpp"

namespace specfem::time_scheme {

/**
 * @brief Base class for time integration schemes
 *
 * Provides the interface for time-stepping algorithms in wave propagation
 * simulations. Handles forward and backward time integration, seismogram
 * output control, and predictor-corrector phases.
 *
 * Derived classes implement specific integration schemes (e.g., `newmark`).
 */
class time_scheme {
public:
  /**
   * @name Constructors
   */
  ///@{

  /**
   * @brief Construct time scheme
   *
   * @param nstep Number of timesteps
   * @param nstep_between_samples Number of timesteps between seismogram samples
   * @param dt Time step
   */
  time_scheme(const int nstep, const int nstep_between_samples,
              const type_real dt)
      : nstep(nstep), nstep_between_samples(nstep_between_samples),
        seismogram_timestep(0), dt(dt) {}
  ///@}

  /**
   * @name Iterators
   */
  ///@{

  /**
   * @brief Forward iterator
   *
   * @return std::tuple<int, type_real> Tuple of current timestep (istep) and
   * time increment (dt)
   *
   * @code
   * /// increments time step
   * for (const auto [istep, dt] : ts.iterate_forward()) {
   *   const auto time = istep * dt; /// Computing the current time
   * }
   * @endcode
   */
  impl::ForwardRange iterate_forward() { return impl::ForwardRange(nstep, dt); }

  /**
   * @brief Backward iterator
   *
   * @return std::tuple<int, type_real> Tuple of current timestep (istep) and
   * time increment (dt)
   *
   * @code
   * /// decrements time step
   * for (const auto [istep, dt] : ts.iterate_backward()) {
   *   const auto time = istep * dt; /// Computing the current time
   * }
   * @endcode
   */
  impl::BackwardRange iterate_backward() {
    return impl::BackwardRange(nstep, dt);
  }
  ///@}

  /**
   * @brief Get the max timestep
   *
   * @return int Maximum number of timesteps
   */
  int get_max_timestep() { return nstep; }

  /**
   * @brief Increment seismogram output step
   */
  void increment_seismogram_step() { seismogram_timestep++; }

  /**
   * @brief Checks if seismogram should be computed at current timestep
   *
   * @param istep Current timestep
   * @return bool True if seismogram should be computed
   */
  bool compute_seismogram(const int istep) const {
    return (istep % nstep_between_samples == 0);
  }

  /**
   * @brief Get the current seismogram step
   *
   * @return int Seismogram timestep
   */
  int get_seismogram_step() const { return seismogram_timestep; }

  /**
   * @brief Get the maximum seismogram step
   *
   * @return int Maximum seismogram step
   */
  int get_max_seismogram_step() const { return nstep / nstep_between_samples; }

  /**
   * @brief Get the number of timesteps between seismogram samples
   *
   * @return int Number of timesteps between seismogram samples
   */
  int get_nstep_between_samples() const { return nstep_between_samples; }

  /**
   * @brief Apply predictor phase for forward time integration
   *
   * Updates displacement and velocity using current acceleration for the
   * forward wavefield. Pure virtual method implemented by derived classes.
   *
   * @param tag Medium type to process (elastic, acoustic, etc.)
   * @return Number of degrees of freedom updated
   */
  virtual int
  apply_predictor_phase_forward(const specfem::element::medium_tag tag) = 0;

  /**
   * @brief Apply corrector phase for forward time integration
   *
   * Updates velocity using newly computed acceleration for the forward
   * wavefield. Pure virtual method implemented by derived classes.
   *
   * @param tag Medium type to process (elastic, acoustic, etc.)
   * @return Number of degrees of freedom updated
   */
  virtual int
  apply_corrector_phase_forward(const specfem::element::medium_tag tag) = 0;

  /**
   * @brief Apply predictor phase for backward time integration
   *
   * Updates displacement and velocity using current acceleration for the
   * backward wavefield. Pure virtual method implemented by derived classes.
   *
   * @param tag Medium type to process (elastic, acoustic, etc.)
   * @return Number of degrees of freedom updated
   */
  virtual int
  apply_predictor_phase_backward(const specfem::element::medium_tag tag) = 0;

  /**
   * @brief Apply corrector phase for backward time integration
   *
   * Updates velocity using newly computed acceleration for the backward
   * wavefield. Pure virtual method implemented by derived classes.
   *
   * @param tag Medium type to process (elastic, acoustic, etc.)
   * @return Number of degrees of freedom updated
   */
  virtual int
  apply_corrector_phase_backward(const specfem::element::medium_tag tag) = 0;

  /**
   * @brief Get the time scheme type
   *
   * @return Type of time integration scheme (e.g., Newmark)
   */
  virtual specfem::enums::time_scheme::type timescheme() const = 0;

  virtual ~time_scheme() = default;

  /**
   * @brief Convert time scheme to string representation
   *
   * @return String describing the time scheme configuration
   */
  virtual std::string to_string() const = 0;

  /**
   * @brief Print time scheme details to output stream
   *
   * @param out Output stream
   */
  virtual void print(std::ostream &out) const = 0;

  /**
   * @brief Get the time increment per step
   *
   * @return Time step size
   */
  virtual type_real get_timestep() const = 0;

private:
  int nstep;                 ///< Number of timesteps
  int seismogram_timestep;   ///< Current seismogram timestep
  int nstep_between_samples; ///< Number of timesteps between seismogram output
                             ///< samples
  type_real dt;              ///< Time increment
};

std::ostream &operator<<(std::ostream &out,
                         specfem::time_scheme::time_scheme &ts);

} // namespace specfem::time_scheme
