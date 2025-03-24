#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct kernels_container<specfem::element::medium_tag::poroelastic,
                         specfem::element::property_tag::isotropic>
    : public impl_kernels_container<specfem::element::medium_tag::poroelastic,
                                    specfem::element::property_tag::isotropic,
                                    19> {
  using base_type =
      impl_kernels_container<specfem::element::medium_tag::poroelastic,
                             specfem::element::property_tag::isotropic, 19>;
  using base_type::base_type;

  DEFINE_MEDIUM_VIEW(rhot, 0)
  DEFINE_MEDIUM_VIEW(rhof, 1)
  DEFINE_MEDIUM_VIEW(eta, 2)
  DEFINE_MEDIUM_VIEW(sm, 3)
  DEFINE_MEDIUM_VIEW(mu_fr, 4)
  DEFINE_MEDIUM_VIEW(H, 5)
  DEFINE_MEDIUM_VIEW(C, 6)
  DEFINE_MEDIUM_VIEW(M, 7)

  /// Density Normalized Kernels
  DEFINE_MEDIUM_VIEW(mu_frb, 8)
  DEFINE_MEDIUM_VIEW(rhob, 9)
  DEFINE_MEDIUM_VIEW(rhofb, 10)
  DEFINE_MEDIUM_VIEW(phi, 11)

  /// wavespeed kernels
  DEFINE_MEDIUM_VIEW(cpI, 12)
  DEFINE_MEDIUM_VIEW(cpII, 13)
  DEFINE_MEDIUM_VIEW(cs, 14)
  DEFINE_MEDIUM_VIEW(rhobb, 15)
  DEFINE_MEDIUM_VIEW(rhofbb, 16)
  DEFINE_MEDIUM_VIEW(ratio, 17)
  DEFINE_MEDIUM_VIEW(phib, 18)
  ///@}
};

} // namespace medium
} // namespace specfem
