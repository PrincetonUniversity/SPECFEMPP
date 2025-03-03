#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct properties_container<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic>
    : public impl::impl_properties_container<
          specfem::element::medium_tag::elastic,
          specfem::element::property_tag::isotropic, 3> {
  using base_type =
      impl::impl_properties_container<specfem::element::medium_tag::elastic,
                                      specfem::element::property_tag::isotropic,
                                      3>;
  using base_type::base_type;
  constexpr static int _counter = __COUNTER__;

  DEFINE_CONTAINER(lambdaplus2mu)
  DEFINE_CONTAINER(mu)
  DEFINE_CONTAINER(rho)
};

} // namespace medium
} // namespace specfem
