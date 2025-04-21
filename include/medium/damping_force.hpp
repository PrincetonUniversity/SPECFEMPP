#pragma once

#include "enumerations/medium.hpp"
#include "medium/dim2/poroelastic/isotropic/damping.hpp"
#include <Kokkos_Core.hpp>

/*
 * There are two function here at the top outside of the namespace because they
 * are used in the specialization of the function below. The first one is the
 * specialization for the case where the damping force is not activated. A false
 * type is passed to the function, and it does nothing. The second one is the
 * specialization for the case where the damping force is activated. A true
 * type is passed to the function, and it computes the damping force. This
 * allows for compilation time checking for supposed existence of the damping
 * term computation.
 */
template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void
execute_damping_force(std::false_type, const T factor,
                      const PointPropertiesType &point_properties,
                      const PointVelocityType &velocity,
                      PointAccelerationType &acceleration) {
  // No damping force
  return;
}

template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void execute_damping_force(
    std::true_type, const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;

  static_assert(std::is_same_v<T, typename PointPropertiesType::simd::datatype>,
                "factor must have the same SIMD type as point_properties");

  static_assert(PointPropertiesType::is_point_properties,
                "point_properties is not a point properties type");

  static_assert(PointVelocityType::isPointFieldType,
                "velocity is not a point field type");

  static_assert(PointAccelerationType::isPointFieldType,
                "acceleration is not a point field type");

  static_assert(PointVelocityType::store_velocity,
                "velocity must store velocity");

  static_assert(PointAccelerationType::store_acceleration,
                "acceleration must store acceleration");

  static_assert(PointPropertiesType::dimension == PointVelocityType::dimension,
                "point_properties and velocity have different dimensions");

  static_assert(PointPropertiesType::dimension ==
                    PointAccelerationType::dimension,
                "point_properties and acceleration have different dimensions");

  static_assert(PointPropertiesType::medium_tag ==
                    PointVelocityType::medium_tag,
                "point_properties and velocity have different medium tags");

  static_assert(PointPropertiesType::medium_tag ==
                    PointAccelerationType::medium_tag,
                "point_properties and acceleration have different medium tags");

  static_assert(PointPropertiesType::simd::using_simd ==
                    PointVelocityType::simd::using_simd,
                "point_properties and velocity have different SIMD settings");

  static_assert(
      PointPropertiesType::simd::using_simd ==
          PointAccelerationType::simd::using_simd,
      "point_properties and acceleration have different SIMD settings");

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type,
                             specfem::dimension::type::dim2>;
  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;
  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  // Damping force
  specfem::medium::impl_compute_damping_force(
      dimension_dispatch(), medium_dispatch(), property_dispatch(), factor,
      point_properties, velocity, acceleration);
}
namespace specfem {
namespace medium {

/**
 * @defgroup MediumPhysics
 */

/**
 * @brief Compute the damping term at a quadrature point
 *
 * @ingroup MediumPhysics
 *
 * @tparam PointPropertiesType Material properties at the quadrature point
 * specfem::point::properties
 * @tparam PointVelocityType Velocity at the quadrature point
 * specfem::point::field
 * @tparam PointAccelerationType Acceleration at the quadrature point
 * specfem::point::field
 * @param factor Prefactor for the damping term ($wx * wz * jacobian)
 * @param point_properties Material properties at the quadrature point
 * @param velocity Velocity at the quadrature point
 * @param acceleration Acceleration at the quadrature point
 */
template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void compute_damping_force(
    const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {

  constexpr auto DimensionTag = PointPropertiesType::dimension;
  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr bool is_activated =
      specfem::element::attributes<DimensionTag, MediumTag>::damping_force;

  using damping_force_activated_dispatch =
      std::integral_constant<bool, is_activated>;

  execute_damping_force(damping_force_activated_dispatch(), factor,
                        point_properties, velocity, acceleration);
}
} // namespace medium
} // namespace specfem
