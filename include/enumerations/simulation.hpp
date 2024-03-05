#ifndef _ENUMERATIONS_SIMULATION_HPP_
#define _ENUMERATIONS_SIMULATION_HPP_

#include "specfem_enums.hpp"

namespace specfem {
namespace simulation {

enum class type { forward, adjoint };

template <specfem::simulation::type SimulationType> class simulation;

template <> class simulation<specfem::simulation::type::forward> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::forward;
  static std::string to_string() { return "Forward"; }
};

template <> class simulation<specfem::simulation::type::adjoint> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::adjoint;
  static std::string to_string() { return "Adjoint"; }
};

} // namespace simulation
} // namespace specfem

#endif /* _ENUMERATIONS_SIMULATION_HPP_ */
