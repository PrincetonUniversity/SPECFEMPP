#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {
namespace impl {
template <specfem::element::medium_tag MediumTag>
struct kernels_container_elastic_anisotropic
    : public impl_kernels_container<
          MediumTag, specfem::element::property_tag::anisotropic, 7> {
  using base_type =
      impl_kernels_container<MediumTag,
                             specfem::element::property_tag::anisotropic, 7>;
  using base_type::base_type;

  DEFINE_MEDIUM_VIEW(rho, 0)
  DEFINE_MEDIUM_VIEW(c11, 1)
  DEFINE_MEDIUM_VIEW(c13, 2)
  DEFINE_MEDIUM_VIEW(c15, 3)
  DEFINE_MEDIUM_VIEW(c33, 4)
  DEFINE_MEDIUM_VIEW(c35, 5)
  DEFINE_MEDIUM_VIEW(c55, 6)
};
} // namespace impl

template <>
struct kernels_container<specfem::element::medium_tag::elastic_sv,
                         specfem::element::property_tag::anisotropic>
    : public impl::kernels_container_elastic_anisotropic<
          specfem::element::medium_tag::elastic_sv> {
  using base_type = impl::kernels_container_elastic_anisotropic<
      specfem::element::medium_tag::elastic_sv>;
  using base_type::base_type;
};

template <>
struct kernels_container<specfem::element::medium_tag::elastic_sh,
                         specfem::element::property_tag::anisotropic>
    : public impl::kernels_container_elastic_anisotropic<
          specfem::element::medium_tag::elastic_sh> {
  using base_type = impl::kernels_container_elastic_anisotropic<
      specfem::element::medium_tag::elastic_sh>;
  using base_type::base_type;
};

} // namespace medium
} // namespace specfem
