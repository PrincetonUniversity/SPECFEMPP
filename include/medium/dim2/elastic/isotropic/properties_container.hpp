#pragma once

#include "element/impl/is_elastic_2d.hpp"
#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag>
struct properties_container<MediumTag,
                            specfem::element::property_tag::isotropic>
    : public impl_properties_container<
          MediumTag, specfem::element::property_tag::isotropic, 3>,
      element::impl::is_elastic_2d<MediumTag> {
  using base_type =
      impl_properties_container<MediumTag,
                                specfem::element::property_tag::isotropic, 3>;
  using base_type::base_type;

  DEFINE_MEDIUM_VIEW(lambdaplus2mu, 0)
  DEFINE_MEDIUM_VIEW(mu, 1)
  DEFINE_MEDIUM_VIEW(rho, 2)
};

} // namespace medium
} // namespace specfem
