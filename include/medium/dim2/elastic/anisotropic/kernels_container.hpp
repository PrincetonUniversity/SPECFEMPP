#pragma once

#include "element/impl/is_elastic_2d.hpp"
#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag>
struct kernels_container<MediumTag, specfem::element::property_tag::anisotropic>
    : public impl_kernels_container<
          MediumTag, specfem::element::property_tag::anisotropic, 7>,
      element::impl::is_elastic_2d<MediumTag> {
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

} // namespace medium
} // namespace specfem
