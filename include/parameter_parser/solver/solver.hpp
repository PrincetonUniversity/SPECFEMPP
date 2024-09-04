#ifndef _SPECFEM_RUNTIME_CONFIGURATION_SOLVER_SOLVER_HPP_
#define _SPECFEM_RUNTIME_CONFIGURATION_SOLVER_SOLVER_HPP_

#include "compute/interface.hpp"
#include "solver/solver.hpp"
#include "timescheme/newmark.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace runtime_configuration {
namespace solver {

class solver {
public:
  solver(const char *simulation_type) : simulation_type(simulation_type) {}
  solver(const std::string simulation_type)
      : simulation_type(simulation_type) {}

  template <typename qp_type>
  std::shared_ptr<specfem::solver::solver>
  instantiate(const type_real dt, const specfem::compute::assembly &assembly,
              std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
              const qp_type &quadrature) const;

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
  std::string simulation_type;
};
} // namespace solver
} // namespace runtime_configuration
} // namespace specfem

#endif /* _SPECFEM_RUNTIME_CONFIGURATION_SOLVER_SOLVER_HPP_ */
