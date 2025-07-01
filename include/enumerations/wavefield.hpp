#pragma once

#include "dimension.hpp"

namespace specfem {
namespace wavefield {
/**
 * @brief Wavefield tag within the simulation
 *
 */
enum class simulation_field { forward, adjoint, backward, buffer };

/**
 * @brief Type of wavefield component
 *
 */
enum class type { displacement, velocity, acceleration, pressure, rotation, intrinsic_rotation, curl};

/**
 * @brief Defines compile time constants for wavefield components
 *
 * @tparam DimensionTag Dimension of the wavefield
 * @tparam Component Type of the wavefield component
 */
template <specfem::dimension::type DimensionTag,
          specfem::wavefield::type Component>
class wavefield;

// clang-format off
/**
 * @fn static constexpr specfem::dimension::type specfem::wavefield::wavefield::dimension()
 * @brief Returns the dimension type of the wavefield
 *
 * @return specfem::dimension::type Dimension type of the wavefield
 * @memberof specfem::wavefield::wavefield
 */

/**
 * @fn static constexpr specfem::wavefield::type specfem::wavefield::wavefield::component()
 *
 * @brief Returns the component type of the wavefield
 *
 * @return specfem::wavefield::type Component type of the wavefield
 * @memberof specfem::wavefield::wavefield
 */

/**
 * @fn static constexpr int specfem::wavefield::wavefield::num_components()
 * @brief Returns the number of components of the wavefield
 *
 * @return int Number of components of the wavefield
 * @memberof specfem::wavefield::wavefield
 */
// clang-format on

template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::type::displacement> {
public:
  static constexpr auto dimension() { return specfem::dimension::type::dim2; }
  static constexpr auto component() {
    return specfem::wavefield::type::displacement;
  }
  static constexpr int num_components() { return 2; }
};

template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::type::velocity> {
public:
  static constexpr auto dimension() { return specfem::dimension::type::dim2; }
  static constexpr auto component() {
    return specfem::wavefield::type::velocity;
  }
  static constexpr int num_components() { return 2; }
};

template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::type::acceleration> {
public:
  static constexpr auto dimension() { return specfem::dimension::type::dim2; }
  static constexpr auto component() {
    return specfem::wavefield::type::acceleration;
  }
  static constexpr int num_components() { return 2; }
};

template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::type::pressure> {
public:
  static constexpr auto dimension() { return specfem::dimension::type::dim2; }
  static constexpr auto component() {
    return specfem::wavefield::type::pressure;
  }
  static constexpr int num_components() { return 1; }
};


/**
 * @brief Gets the string representation of a wavefield component
 * 
 * @param wavefield_component The wavefield component to convert
 * @return std::string The string representation of the wavefield component
 *
 */
const std::string to_string(const specfem::wavefield::type &wavefield_component);

} // namespace wavefield
} // namespace specfem
