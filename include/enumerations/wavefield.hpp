#pragma once

#include "dimension.hpp"

namespace specfem {
namespace wavefield {
/**
 * @brief Wavefield type enumeration
 *
 */
enum class type { forward, adjoint, backward, buffer };

enum class component { displacement, velocity, acceleration };

template <specfem::dimension::type DimensionType,
          specfem::wavefield::component Component>
class wavefield;

template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::component::displacement> {
public:
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto component = specfem::wavefield::component::displacement;
  static constexpr int num_components = 2;
};

template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::component::velocity> {
public:
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto component = specfem::wavefield::component::velocity;
  static constexpr int num_components = 2;
};

template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::component::acceleration> {
public:
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto component = specfem::wavefield::component::acceleration;
  static constexpr int num_components = 2;
};

} // namespace wavefield
} // namespace specfem
