#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {
namespace impl {
template <specfem::element::medium_tag MediumTag>
struct kernels_container_elastic_isotropic
    : public impl_kernels_container<
          MediumTag, specfem::element::property_tag::isotropic, 6> {
  using base_type =
      impl_kernels_container<MediumTag,
                             specfem::element::property_tag::isotropic, 6>;
  using base_type::base_type;

  DEFINE_MEDIUM_VIEW(rho, 0)
  DEFINE_MEDIUM_VIEW(mu, 1)
  DEFINE_MEDIUM_VIEW(kappa, 2)
  DEFINE_MEDIUM_VIEW(rhop, 3)
  DEFINE_MEDIUM_VIEW(alpha, 4)
  DEFINE_MEDIUM_VIEW(beta, 5)
};
} // namespace impl

template <>
struct kernels_container<specfem::element::medium_tag::elastic_sv,
                         specfem::element::property_tag::isotropic>
    : public impl::kernels_container_elastic_isotropic<
          specfem::element::medium_tag::elastic_sv> {
  using base_type = impl::kernels_container_elastic_isotropic<
      specfem::element::medium_tag::elastic_sv>;
  using base_type::base_type;
};

template <>
struct kernels_container<specfem::element::medium_tag::elastic_sh,
                         specfem::element::property_tag::isotropic>
    : public impl::kernels_container_elastic_isotropic<
          specfem::element::medium_tag::elastic_sh> {
  using base_type = impl::kernels_container_elastic_isotropic<
      specfem::element::medium_tag::elastic_sh>;
  using base_type::base_type;
};

} // namespace medium
} // namespace specfem
