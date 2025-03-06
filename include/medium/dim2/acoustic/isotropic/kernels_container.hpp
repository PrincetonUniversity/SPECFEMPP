#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct kernels_container<specfem::element::medium_tag::acoustic,
                         specfem::element::property_tag::isotropic>
    : public impl_kernels_container<specfem::element::medium_tag::acoustic,
                                    specfem::element::property_tag::isotropic,
                                    4> {
  using base_type =
      impl_kernels_container<specfem::element::medium_tag::acoustic,
                             specfem::element::property_tag::isotropic, 4>;
  using base_type::base_type;
  constexpr static int _counter = __COUNTER__;

  DEFINE_MEDIUM_VIEW(rho)
  DEFINE_MEDIUM_VIEW(kappa)
  DEFINE_MEDIUM_VIEW(rhop)
  DEFINE_MEDIUM_VIEW(alpha)
};

} // namespace medium
} // namespace specfem
