#pragma once

#include "dimension.hpp"
#include <string>

namespace specfem {
namespace wavefield {
/**
 * @brief Simulation field types for wave propagation algorithms.
 *
 * Used in time-stepping schemes and inversion methods.
 */
enum class simulation_field {
  forward,  ///< Forward time propagation
  adjoint,  ///< Adjoint field (backward from receivers)
  backward, ///< Backward field (for gradient computation)
  buffer    ///< Temporary buffer field
};

/**
 * @brief Wavefield component types for different physical quantities.
 *
 * Supports elastic (displacement/velocity/acceleration) and acoustic (pressure)
 * fields.
 */
enum class type {
  displacement,       ///< Displacement field (elastic media)
  velocity,           ///< Velocity field (time derivative of displacement)
  acceleration,       ///< Acceleration field (second time derivative)
  pressure,           ///< Pressure field (acoustic media)
  rotation,           ///< Rotation field (Cosserat media)
  intrinsic_rotation, ///< Intrinsic rotation (micropolar)
  curl                ///< Curl of displacement field
};

/**
 * @brief Compile-time wavefield component traits.
 *
 * Provides dimension-specific component counts and type information.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam Component Wavefield component type
 *
 * @code
 * using disp_2d = wavefield<dim2, type::displacement>;
 * static_assert(disp_2d::num_components() == 2); // u_x, u_z
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::wavefield::type Component>
class wavefield;

// Specializations provide:
// - dimension(): spatial dimension type
// - component(): wavefield component type
// - num_components(): number of field components

/**
 * @brief 2D displacement wavefield (u_x, u_z).
 */
template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::type::displacement> {
public:
  static constexpr auto dimension() {
    return specfem::dimension::type::dim2;
  } ///< 2D dimension
  static constexpr auto component() {
    return specfem::wavefield::type::displacement;
  } ///< Displacement component type
  static constexpr int num_components() { return 2; } ///< u_x, u_z components
};

/**
 * @brief 2D velocity wavefield (v_x, v_z).
 */
template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::type::velocity> {
public:
  static constexpr auto dimension() {
    return specfem::dimension::type::dim2;
  } ///< 2D dimension
  static constexpr auto component() {
    return specfem::wavefield::type::velocity;
  } ///< Velocity component type
  static constexpr int num_components() { return 2; } ///< v_x, v_z components
};

/**
 * @brief 2D acceleration wavefield (a_x, a_z).
 */
template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::type::acceleration> {
public:
  static constexpr auto dimension() {
    return specfem::dimension::type::dim2;
  } ///< 2D dimension
  static constexpr auto component() {
    return specfem::wavefield::type::acceleration;
  } ///< Acceleration component type
  static constexpr int num_components() { return 2; } ///< a_x, a_z components
};

/**
 * @brief 2D pressure wavefield (scalar acoustic field).
 */
template <>
class wavefield<specfem::dimension::type::dim2,
                specfem::wavefield::type::pressure> {
public:
  static constexpr auto dimension() {
    return specfem::dimension::type::dim2;
  } ///< 2D dimension
  static constexpr auto component() {
    return specfem::wavefield::type::pressure;
  } ///< Pressure component type
  static constexpr int num_components() {
    return 1;
  } ///< Scalar pressure component
};

/**
 * @brief Convert wavefield component to string.
 * @param wavefield_component Wavefield component type
 * @return String representation ("displacement", "velocity", etc.)
 */
const std::string
to_string(const specfem::wavefield::type &wavefield_component);

} // namespace wavefield
} // namespace specfem
