#pragma once
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace specfem {

/**
 * @brief Simulation configuration and type management for SPECFEM computations.
 *
 * Provides simulation types (forward, combined), field types for time-stepping,
 * computational models, and type traits. Includes string conversion utilities
 * for parameter parsing and configuration.
 *
 */
namespace simulation {

/**
 * @brief Simulation execution types.
 */
enum class type {
  forward,         ///< Forward simulation only
  combined,        ///< Combined forward and adjoint simulation
  combined_undoatt ///< Adjoint with exact forward replay (UNDO_ATTENUATION)
};

/**
 * @brief Simulation field types for wave propagation algorithms.
 *
 * Used in time-stepping schemes and inversion methods.
 */
enum class field_type {
  forward,  ///< Forward time propagation
  adjoint,  ///< Adjoint field (backward from receivers)
  backward, ///< Backward field (for gradient computation)
  buffer    ///< Temporary buffer field
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
 * @brief Convert field type to string.
 *
 * @param field Field type
 * @return String representation
 */
inline std::string to_string(specfem::simulation::field_type field) {
  switch (field) {
  case specfem::simulation::field_type::forward:
    return "forward";
  case specfem::simulation::field_type::adjoint:
    return "adjoint";
  case specfem::simulation::field_type::backward:
    return "backward";
  case specfem::simulation::field_type::buffer:
    return "buffer";
  default:
    return "unknown";
  }
}

/**
 * @brief Simulation type traits template.
 * @tparam SimulationType Simulation type (forward, combined, or
 * combined_undoatt)
 */
template <specfem::simulation::type SimulationType> class simulation;

/**
 * @brief Forward simulation type traits.
 */
template <> class simulation<specfem::simulation::type::forward> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::forward;
};

/**
 * @brief Combined simulation type traits.
 */
template <> class simulation<specfem::simulation::type::combined> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::combined;
};

/**
 * @brief Undo-attenuation simulation type traits.
 *
 * Runs a full forward pass (with checkpointing) followed by an adjoint pass
 * in which the forward wavefield is replayed exactly from saved snapshots,
 * matching the Fortran UNDO_ATTENUATION_AND_OR_PML strategy.
 */
template <> class simulation<specfem::simulation::type::combined_undoatt> {
public:
  static constexpr auto simulation_type =
      specfem::simulation::type::combined_undoatt;
};

} // namespace simulation
} // namespace specfem
