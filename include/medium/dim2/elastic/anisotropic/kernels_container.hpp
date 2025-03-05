#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct kernels_container<specfem::element::medium_tag::elastic,
                         specfem::element::property_tag::anisotropic>
    : public impl_kernels_container<specfem::element::medium_tag::elastic,
                                    specfem::element::property_tag::anisotropic,
                                    7> {
  using base_type =
      impl_kernels_container<specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::anisotropic, 7>;
  using base_type::base_type;
  constexpr static int _counter = __COUNTER__;

  DEFINE_MEDIUM_VIEW(rho)
  DEFINE_MEDIUM_VIEW(c11)
  DEFINE_MEDIUM_VIEW(c13)
  DEFINE_MEDIUM_VIEW(c15)
  DEFINE_MEDIUM_VIEW(c33)
  DEFINE_MEDIUM_VIEW(c35)
  DEFINE_MEDIUM_VIEW(c55)
};

} // namespace medium
} // namespace specfem
