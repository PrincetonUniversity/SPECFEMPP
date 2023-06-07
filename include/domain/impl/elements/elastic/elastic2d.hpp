#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_HPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_HPP

#include "domain/impl/elements/element.hpp"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace Domain {
namespace impl {
namespace elements {
template <typename quadrature_points>
class element<specfem::enums::element::dimension::dim2, quadrature_points,
              specfem::enums::element::medium::elastic> {

public:
  KOKKOS_FUNCTION virtual void compute_gradient(
      const int &xz,
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          s_hprime_xx,
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          s_hprime_zz,
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          field_x,
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          field_z,
      type_real &duxdxl, type_real &duxdzl, type_real &duzdxl,
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
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          stress_integrand_1,
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          stress_integrand_2,
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          stress_integrand_3,
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          stress_integrand_4,
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          s_hprimewgll_xx,
      const specfem::kokkos::StaticDeviceScratchView2d<
          type_real, quadrature_points::NGLL, quadrature_points::NGLL>
          s_hprimewgll_zz) const {};
}; // namespace element

template <>
class element<specfem::enums::element::dimension::dim2,
              specfem::enums::element::medium::elastic> {

public:
  KOKKOS_FUNCTION virtual void compute_gradient(
      const int &xz,
      const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_xx,
      const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_zz,
      const specfem::kokkos::DeviceScratchView2d<type_real> field_x,
      const specfem::kokkos::DeviceScratchView2d<type_real> field_z,
      type_real &duxdxl, type_real &duxdzl, type_real &duzdxl,
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
      const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_1,
      const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_2,
      const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_3,
      const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_4,
      const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_xx,
      const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_zz)
      const {};

}; // namespace element

} // namespace elements
} // namespace impl
} // namespace Domain
} // namespace specfem

#endif
