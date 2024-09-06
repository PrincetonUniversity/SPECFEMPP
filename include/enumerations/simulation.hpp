#pragma once

namespace specfem {
namespace simulation {
/**
 * @brief Simulation type enumeration
 *
 */
enum class type { forward, combined };

template <specfem::simulation::type SimulationType> class simulation;

template <> class simulation<specfem::simulation::type::forward> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::forward;
  static std::string to_string() { return "Forward"; }
};

template <> class simulation<specfem::simulation::type::combined> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::combined;
  static std::string to_string() { return "Adjoint & Forward combined"; }
};

} // namespace simulation
} // namespace specfem
