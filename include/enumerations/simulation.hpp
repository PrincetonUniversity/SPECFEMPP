#pragma once
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace specfem {
namespace simulation {

/**
 * @brief Simulation execution types.
 */
enum class type {
  forward, ///< Forward simulation only
  combined ///< Combined forward and adjoint simulation
};

/**
 * @brief Simulation model enumeration
 *
 * Defines the different simulation models supported by SPECFEM++.
 * Currently supports Cartesian 2D and 3D simulations.
 * Future models: Axisem2D (axisymmetric) and Globe3D (global spherical)
 */
enum class model {
  Cartesian2D,
  Cartesian3D
  // Axisem2D,   // Not yet implemented - axisymmetric 2D
  // Globe3D     // Not yet implemented - global 3D spherical
};

/**
 * @brief Convert string to simulation model
 *
 * @param str String representation of simulation model
 * @return specfem::simulation::model Corresponding simulation model
 * @throws std::invalid_argument if string is not a valid simulation model
 */
inline model from_string(const std::string &str) {
  static const std::unordered_map<std::string, specfem::simulation::model>
      models = {
        { "2d", specfem::simulation::model::Cartesian2D },
        { "dim2", specfem::simulation::model::Cartesian2D },
        { "cartesian2d", specfem::simulation::model::Cartesian2D },
        { "3d", specfem::simulation::model::Cartesian3D },
        { "dim3", specfem::simulation::model::Cartesian3D },
        { "cartesian3d", specfem::simulation::model::Cartesian3D }
        // { "axisem2d", specfem::simulation::model::Axisem2D },  // Not yet
        // implemented { "globe3d", specfem::simulation::model::Globe3D }     //
        // Not yet implemented
      };

  auto it = models.find(str);
  if (it == models.end()) {
    throw std::invalid_argument("Invalid simulation model: " + str +
                                ". Use '2d', 'dim2', '3d', or 'dim3'.");
  }
  return it->second;
}

/**
 * @brief Convert simulation model to string
 *
 * @param mdl Simulation model
 * @return std::string String representation
 */
inline std::string to_string(specfem::simulation::model mdl) {
  switch (mdl) {
  case specfem::simulation::model::Cartesian2D:
    return "Cartesian 2D";
  case specfem::simulation::model::Cartesian3D:
    return "Cartesian 3D";
  // case specfem::simulation::model::Axisem2D:    // Not yet implemented
  //   return "Axisymmetric 2D";
  // case specfem::simulation::model::Globe3D:     // Not yet implemented
  //   return "Global 3D";
  default:
    return "Unknown";
  }
}

/**
 * @brief Simulation type traits template.
 * @tparam SimulationType Simulation type (forward or combined)
 */
template <specfem::simulation::type SimulationType> class simulation;

/**
 * @brief Forward simulation type traits.
 */
template <> class simulation<specfem::simulation::type::forward> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::forward;

  /**
   * @brief Get simulation type name.
   * @return String representation of simulation type
   */
  static std::string to_string() { return "Forward"; }
};

/**
 * @brief Combined simulation type traits.
 */
template <> class simulation<specfem::simulation::type::combined> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::combined;

  /**
   * @brief Get simulation type name.
   * @return String representation of simulation type
   */
  static std::string to_string() { return "Adjoint & Forward combined"; }
};

} // namespace simulation
} // namespace specfem
