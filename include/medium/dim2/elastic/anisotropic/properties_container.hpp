#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct properties_container<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::anisotropic>
    : public impl_properties_container<
          specfem::element::medium_tag::elastic,
          specfem::element::property_tag::anisotropic, 10> {
  using base_type =
      impl_properties_container<specfem::element::medium_tag::elastic,
                                specfem::element::property_tag::anisotropic,
                                10>;
  using base_type::base_type;
  constexpr static int _counter = __COUNTER__;

  DEFINE_MEDIUM_VIEW(c11)
  DEFINE_MEDIUM_VIEW(c13)
  DEFINE_MEDIUM_VIEW(c15)
  DEFINE_MEDIUM_VIEW(c33)
  DEFINE_MEDIUM_VIEW(c35)
  DEFINE_MEDIUM_VIEW(c55)
  DEFINE_MEDIUM_VIEW(c12)
  DEFINE_MEDIUM_VIEW(c23)
  DEFINE_MEDIUM_VIEW(c25)
  DEFINE_MEDIUM_VIEW(rho)
};

} // namespace medium
} // namespace specfem
