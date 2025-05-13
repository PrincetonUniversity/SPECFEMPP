#pragma once

#include "medium/properties_container.hpp"
#include "specfem/point.hpp"
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

  DEFINE_MEDIUM_VIEW(rho_inverse, 0)
  DEFINE_MEDIUM_VIEW(kappa, 1)
};

} // namespace medium
} // namespace specfem
