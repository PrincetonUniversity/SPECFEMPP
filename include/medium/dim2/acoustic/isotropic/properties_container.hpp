#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct properties_container<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic>
    : public impl_properties_container<
          specfem::element::medium_tag::acoustic,
          specfem::element::property_tag::isotropic, 2> {
  using base_type =
      impl_properties_container<specfem::element::medium_tag::acoustic,
                                specfem::element::property_tag::isotropic, 2>;
  using base_type::base_type;
  constexpr static int _counter = __COUNTER__;

  DEFINE_MEDIUM_VIEW(rho_inverse)
  DEFINE_MEDIUM_VIEW(kappa)
};

} // namespace medium
} // namespace specfem
