#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_HPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_HPP

#include "domain/impl/elements/element.hpp"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
template <typename quadrature_points>
class element<specfem::enums::element::dimension::dim2,
              specfem::enums::element::medium::elastic, quadrature_points> {

public:
  template <typename T>
  using ScratchViewType =
      typename quadrature_points::template ScratchViewType<T>;

  KOKKOS_FUNCTION virtual void
  compute_gradient(const int &xz, const ScratchViewType<type_real> s_hprime_xx,
                   const ScratchViewType<type_real> s_hprime_zz,
                   const ScratchViewType<type_real> field_x,
                   const ScratchViewType<type_real> field_z, type_real &duxdxl,
                   type_real &duxdzl, type_real &duzdxl,
                   type_real &duzdzl) const {};

  KOKKOS_FUNCTION virtual void
  compute_stress(const int &xz, const type_real &duxdxl,
                 const type_real &duxdzl, const type_real &duzdxl,
                 const type_real &duzdzl, type_real &stress_integrand_1l,
                 type_real &stress_integrand_2l, type_real &stress_integrand_3l,
                 type_real &stress_integrand_4l) const {};

  KOKKOS_FUNCTION virtual void update_acceleration(
      const int &xz, const int &iglob, const type_real &wxglll,
      const type_real &wzglll,
      const ScratchViewType<type_real> stress_integrand_1,
      const ScratchViewType<type_real> stress_integrand_2,
      const ScratchViewType<type_real> stress_integrand_3,
      const ScratchViewType<type_real> stress_integrand_4,
      const ScratchViewType<type_real> s_hprimewgll_xx,
      const ScratchViewType<type_real> s_hprimewgll_zz) const {};

  KOKKOS_FUNCTION virtual int get_ispec() const { return 0; };
}; // namespace element

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
