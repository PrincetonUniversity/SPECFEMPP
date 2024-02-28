#ifndef _ENUMERATIONS_SIMULATION_HPP_
#define _ENUMERATIONS_SIMULATION_HPP_

#include "specfem_enums.hpp"

namespace specfem {
namespace enums {
namespace simulation {
/**
 * @brief Forward simulation
 *
 */
class forward {
public:
  /**
   * @brief constexpr defining the type of the simulation
   *
   */
  constexpr static specfem::enums::simulation::type value =
      specfem::enums::simulation::type::forward;
  /**
   * @brief Convert the simulation to a string
   *
   */
  inline static std::string to_string() { return "Forward"; }
};

/**
 * @brief Adjoint simulation
 *
 */
class adjoint {
public:
  /**
   * @brief constexpr defining the type of the simulation
   *
   */
  constexpr static specfem::enums::simulation::type value =
      specfem::enums::simulation::type::adjoint;
  /**
   * @brief Convert the simulation to a string
   *
   */
  inline static std::string to_string() { return "Adjoint"; }
};
} // namespace simulation
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_SIMULATION_HPP_ */
