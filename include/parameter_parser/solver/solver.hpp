#pragma once

#include "compute/interface.hpp"
#include "periodic_tasks/periodic_task.hpp"
#include "solver/solver.hpp"
#include "timescheme/newmark.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace runtime_configuration {
namespace solver {
/**
 * @brief Solver class to instantiate the correct solver based on the simulation
 * parameters
 *
 */
class solver {
public:
  /**
   * @brief Construct a new solver object
   *
   * @param simulation_type Type of the simulation (forward or combined)
   */
  solver(const char *simulation_type) : simulation_type(simulation_type) {}
  /**
   * @brief Construct a new solver object
   *
   * @param simulation_type Type of the simulation (forward or combined)
   */
  solver(const std::string simulation_type)
      : simulation_type(simulation_type) {}

  /**
   * @brief Instantiate the solver based on the simulation parameters
   *
   * @tparam qp_type Quadrature points type defining compile time or runtime
   * quadrature points
   * @param dt Time step
   * @param assembly Assembly object
   * @param time_scheme Time scheme object
   * @param quadrature Quadrature points object
   * @return std::shared_ptr<specfem::solver::solver> Solver object
   */
  template <int NGLL>
  std::shared_ptr<specfem::solver::solver>
  instantiate(const type_real dt, const specfem::compute::assembly &assembly,
              std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
              const std::vector<
                  std::shared_ptr<specfem::periodic_tasks::periodic_task> >
                  &tasks) const;

  /**
   * @brief Get the type of the simulation (forward or combined)
   *
   * @return specfem::simulation::type Type of the simulation
   */
  inline specfem::simulation::type get_simulation_type() const {
    if (this->simulation_type == "forward") {
      return specfem::simulation::type::forward;
    } else if (this->simulation_type == "combined") {
      return specfem::simulation::type::combined;
    } else {
      throw std::runtime_error("Unknown simulation type");
    }
  }

private:
  std::string simulation_type; ///< Type of the simulation (forward or
                               ///< combined)
};
} // namespace solver
} // namespace runtime_configuration
} // namespace specfem
