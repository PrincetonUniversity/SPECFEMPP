#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct properties_container<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::anisotropic>
    : public impl::impl_properties_container<
          specfem::element::medium_tag::elastic,
          specfem::element::property_tag::anisotropic, 10> {
  using Base = impl::impl_properties_container<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic, 10>;
  using Base::Base;
  constexpr static int _counter = __COUNTER__;

  DEFINE_CONTAINER(c11)
  DEFINE_CONTAINER(c13)
  DEFINE_CONTAINER(c15)
  DEFINE_CONTAINER(c33)
  DEFINE_CONTAINER(c35)
  DEFINE_CONTAINER(c55)
  DEFINE_CONTAINER(c12)
  DEFINE_CONTAINER(c23)
  DEFINE_CONTAINER(c25)
  DEFINE_CONTAINER(rho)
};

} // namespace medium
} // namespace specfem
