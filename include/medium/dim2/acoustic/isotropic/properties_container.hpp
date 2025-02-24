#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct properties_container<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic>
    : public impl::impl_properties_container<
          specfem::element::medium_tag::acoustic,
          specfem::element::property_tag::isotropic, 2> {
  using Base =
      impl::impl_properties_container<specfem::element::medium_tag::acoustic,
                                      specfem::element::property_tag::isotropic,
                                      2>;
  using Base::Base;
  constexpr static int _counter = __COUNTER__;

  DEFINE_CONTAINER(rho_inverse)
  DEFINE_CONTAINER(kappa)
};

} // namespace medium
} // namespace specfem
