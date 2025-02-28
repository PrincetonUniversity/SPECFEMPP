#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct kernels_container<specfem::element::medium_tag::elastic,
                         specfem::element::property_tag::isotropic>
    : public impl_kernels_container<specfem::element::medium_tag::elastic,
                                    specfem::element::property_tag::isotropic,
                                    6> {
  using base_type =
      impl_kernels_container<specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::isotropic, 6>;
  using base_type::base_type;
  constexpr static int _counter = __COUNTER__;

  DEFINE_MEDIUM_VIEW(rho)
  DEFINE_MEDIUM_VIEW(mu)
  DEFINE_MEDIUM_VIEW(kappa)
  DEFINE_MEDIUM_VIEW(rhop)
  DEFINE_MEDIUM_VIEW(alpha)
  DEFINE_MEDIUM_VIEW(beta)
};

} // namespace medium
} // namespace specfem
