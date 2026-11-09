#pragma once

#include "specfem/injection/method/pointwise_surface_method.hpp"
#include <string_view>

namespace specfem::injection_method_test {

/**
 * @brief Test injection method: scale * incident value * dS.
 *
 * Elastic/isotropic points request displacement then velocity and use only the
 * velocity, so its slot offset (3) is exercised; acoustic/isotropic points
 * request the potential rate.
 */
struct ConstantForce
    : specfem::injection::PointwiseSurfaceMethod<ConstantForce> {
  static constexpr std::string_view method_name = "constant-force";

  struct Parameters {
    type_real scale = 2.0;
  };

  using elastic_quantities =
      specfem::injection::QuantityList<specfem_quantity_displacement,
                                       specfem_quantity_velocity>;
  using acoustic_quantities =
      specfem::injection::QuantityList<specfem_quantity_potential_rate>;

  using ElasticContext = Context<specfem::element::medium_tag::elastic,
                                 specfem::element::property_tag::isotropic>;
  using AcousticContext = Context<specfem::element::medium_tag::acoustic,
                                  specfem::element::property_tag::isotropic>;

  KOKKOS_INLINE_FUNCTION static specfem::point::acceleration<
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic, false>>
  compute(const ElasticContext &context) {
    const auto velocity = context.incident<specfem_quantity_velocity>();
    const type_real factor = context.parameters().scale * context.weight();
    return { factor * velocity(0), factor * velocity(1), factor * velocity(2) };
  }

  KOKKOS_INLINE_FUNCTION static specfem::point::acceleration<
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic, false>>
  compute(const AcousticContext &context) {
    const auto rate = context.incident<specfem_quantity_potential_rate>();
    specfem::point::acceleration<
        specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                            specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic, false>>
        acceleration;
    acceleration(0) = context.parameters().scale * context.weight() * rate(0);
    return acceleration;
  }
};

} // namespace specfem::injection_method_test
