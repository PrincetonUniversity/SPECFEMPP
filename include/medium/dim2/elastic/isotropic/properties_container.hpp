#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct properties_container<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic>
    : public impl_properties_container<
          specfem::element::medium_tag::elastic,
          specfem::element::property_tag::isotropic, 3> {
  using base_type =
      impl_properties_container<specfem::element::medium_tag::elastic,
                                specfem::element::property_tag::isotropic, 3>;
  using base_type::base_type;
  constexpr static int _counter = __COUNTER__;

  DEFINE_MEDIUM_VIEW(lambdaplus2mu)
  DEFINE_MEDIUM_VIEW(mu)
  DEFINE_MEDIUM_VIEW(rho)
};

} // namespace medium
} // namespace specfem
