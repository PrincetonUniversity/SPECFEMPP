#pragma once

#include "specfem/injection/method/point_context.hpp"
#include "specfem/injection/method/quantity_list.hpp"
#include "specfem/point.hpp"
#include <string_view>

// Injection methods that each break one requirement of
// PointwiseSurfaceMethod. Only the compile-failure tests wrap them in it.

namespace specfem::injection_method_test {

using ElasticAcceleration = specfem::point::acceleration<
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic, false>>;

using VelocityList =
    specfem::injection::QuantityList<specfem_quantity_velocity>;

struct NoName : specfem::injection::PointwiseMethodBase<NoName> {
  using elastic_quantities = VelocityList;
  KOKKOS_INLINE_FUNCTION static ElasticAcceleration
  compute(const Context<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> &) {
    return {};
  }
};

struct NonStaticCompute
    : specfem::injection::PointwiseMethodBase<NonStaticCompute> {
  static constexpr std::string_view method_name = "non-static-compute";
  using elastic_quantities = VelocityList;
  KOKKOS_INLINE_FUNCTION ElasticAcceleration
  compute(const Context<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> &) const {
    return {};
  }
};

struct MutableRefCompute
    : specfem::injection::PointwiseMethodBase<MutableRefCompute> {
  static constexpr std::string_view method_name = "mutable-ref-compute";
  using elastic_quantities = VelocityList;
  KOKKOS_INLINE_FUNCTION static ElasticAcceleration
  compute(Context<specfem::element::medium_tag::elastic,
                  specfem::element::property_tag::isotropic> &) {
    return {};
  }
};

struct QuantitiesWithoutCompute
    : specfem::injection::PointwiseMethodBase<QuantitiesWithoutCompute> {
  static constexpr std::string_view method_name = "quantities-without-compute";
  using elastic_quantities = VelocityList;
  using acoustic_quantities =
      specfem::injection::QuantityList<specfem_quantity_potential_rate>;
  KOKKOS_INLINE_FUNCTION static ElasticAcceleration
  compute(const Context<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> &) {
    return {};
  }
};

struct ComputeWithoutQuantities
    : specfem::injection::PointwiseMethodBase<ComputeWithoutQuantities> {
  static constexpr std::string_view method_name = "compute-without-quantities";
  KOKKOS_INLINE_FUNCTION static ElasticAcceleration
  compute(const Context<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> &) {
    return {};
  }
};

struct DuplicateQuantities
    : specfem::injection::PointwiseMethodBase<DuplicateQuantities> {
  static constexpr std::string_view method_name = "duplicate-quantities";
  using elastic_quantities =
      specfem::injection::QuantityList<specfem_quantity_velocity,
                                       specfem_quantity_velocity>;
  KOKKOS_INLINE_FUNCTION static ElasticAcceleration
  compute(const Context<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> &) {
    return {};
  }
};

struct UnsupportedQuantity
    : specfem::injection::PointwiseMethodBase<UnsupportedQuantity> {
  static constexpr std::string_view method_name = "unsupported-quantity";
  using elastic_quantities =
      specfem::injection::QuantityList<specfem_quantity_potential_rate>;
  KOKKOS_INLINE_FUNCTION static ElasticAcceleration
  compute(const Context<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> &) {
    return {};
  }
};

struct WrongReturn : specfem::injection::PointwiseMethodBase<WrongReturn> {
  static constexpr std::string_view method_name = "wrong-return";
  using elastic_quantities = VelocityList;
  KOKKOS_INLINE_FUNCTION static type_real
  compute(const Context<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> &) {
    return 0;
  }
};

struct NonTrivialParameters
    : specfem::injection::PointwiseMethodBase<NonTrivialParameters> {
  static constexpr std::string_view method_name = "non-trivial-parameters";
  struct Parameters {
    Parameters() = default;
    Parameters(const Parameters &) {}
  };
  using elastic_quantities = VelocityList;
  KOKKOS_INLINE_FUNCTION static ElasticAcceleration
  compute(const Context<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> &) {
    return {};
  }
};

struct NoDefaultParameters
    : specfem::injection::PointwiseMethodBase<NoDefaultParameters> {
  static constexpr std::string_view method_name = "no-default-parameters";
  struct Parameters {
    explicit Parameters(type_real scale) : scale(scale) {}
    type_real scale;
  };
  using elastic_quantities = VelocityList;
  KOKKOS_INLINE_FUNCTION static ElasticAcceleration
  compute(const Context<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> &) {
    return {};
  }
};

} // namespace specfem::injection_method_test
