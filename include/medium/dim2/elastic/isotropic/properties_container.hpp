#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {
namespace impl {
template <specfem::element::medium_tag MediumTag>
struct properties_container_elastic_isotropic
    : public impl_properties_container<
          MediumTag, specfem::element::property_tag::isotropic, 3> {
  using base_type =
      impl_properties_container<MediumTag,
                                specfem::element::property_tag::isotropic, 3>;
  using base_type::base_type;

  DEFINE_MEDIUM_VIEW(lambdaplus2mu, 0)
  DEFINE_MEDIUM_VIEW(mu, 1)
  DEFINE_MEDIUM_VIEW(rho, 2)
};
} // namespace impl

template <>
struct properties_container<specfem::element::medium_tag::elastic_sv,
                            specfem::element::property_tag::isotropic>
    : public impl::properties_container_elastic_isotropic<
          specfem::element::medium_tag::elastic_sv> {
  using base_type = impl::properties_container_elastic_isotropic<
      specfem::element::medium_tag::elastic_sv>;
  using base_type::base_type;
};

template <>
struct properties_container<specfem::element::medium_tag::elastic_sh,
                            specfem::element::property_tag::isotropic>
    : public impl::properties_container_elastic_isotropic<
          specfem::element::medium_tag::elastic_sh> {
  using base_type = impl::properties_container_elastic_isotropic<
      specfem::element::medium_tag::elastic_sh>;
  using base_type::base_type;
};

} // namespace medium
} // namespace specfem
