#ifndef _ENUMERATIONS_SIMULATION_HPP_
#define _ENUMERATIONS_SIMULATION_HPP_

#include "specfem_enums.hpp"

namespace specfem {
namespace simulation {

enum class type { forward, adjoint };

template <specfem::enums::simulation::type simulation> class simulation {
public:
  static constexpr specfem::enums::simulation::type value = simulation;
  static std::string to_string(){};
};

template <>
std::string simulation<specfem::enums::simulation::type::forward>::to_string() {
  return "Forward";
}

template <>
std::string simulation<specfem::enums::simulation::type::adjoint>::to_string() {
  return "Adjoint";
}

} // namespace simulation
} // namespace specfem

#endif /* _ENUMERATIONS_SIMULATION_HPP_ */
