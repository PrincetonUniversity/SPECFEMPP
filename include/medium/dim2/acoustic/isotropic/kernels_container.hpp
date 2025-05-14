#pragma once

#include "medium/properties_container.hpp"
#include "specfem/point.hpp"
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

  DEFINE_MEDIUM_VIEW(rho, 0)
  DEFINE_MEDIUM_VIEW(kappa, 1)
  DEFINE_MEDIUM_VIEW(rhop, 2)
  DEFINE_MEDIUM_VIEW(alpha, 3)
};

} // namespace medium
} // namespace specfem
