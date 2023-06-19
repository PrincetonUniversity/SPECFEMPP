#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_HPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_ISOTROPIC_HPP

#include "compute/interface.hpp"
#include "domain/impl/elements/elastic/elastic2d.hpp"
#include "domain/impl/elements/element.hpp"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
template <int N>
class element<specfem::enums::element::dimension::dim2,
              specfem::enums::element::medium::elastic,
              specfem::enums::element::quadrature::static_quadrature_points<N>,
              specfem::enums::element::property::isotropic>
    : public element<
          specfem::enums::element::dimension::dim2,
          specfem::enums::element::medium::elastic,
          specfem::enums::element::quadrature::static_quadrature_points<N> > {
public:
  using quadrature_points =
      specfem::enums::element::quadrature::static_quadrature_points<N>;

  template <typename T>
  using ScratchViewType =
      typename quadrature_points::template ScratchViewType<T>;

  KOKKOS_FUNCTION
  element() = default;

  KOKKOS_FUNCTION
  element(const int ispec,
          const specfem::compute::partial_derivatives partial_derivatives,
          const specfem::compute::properties properties,
          const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
              field_dot_dot);

  KOKKOS_FUNCTION void
  compute_gradient(const int &xz, const ScratchViewType<type_real> s_hprime_xx,
                   const ScratchViewType<type_real> s_hprime_zz,
                   const ScratchViewType<type_real> field_x,
                   const ScratchViewType<type_real> field_z, type_real &duxdxl,
                   type_real &duxdzl, type_real &duzdxl,
                   type_real &duzdzl) const override;

  KOKKOS_FUNCTION void
  compute_stress(const int &xz, const type_real &duxdxl,
                 const type_real &duxdzl, const type_real &duzdxl,
                 const type_real &duzdzl, type_real &stress_integrand_1l,
                 type_real &stress_integrand_2l, type_real &stress_integrand_3l,
                 type_real &stress_integrand_4l) const override;

  KOKKOS_FUNCTION void update_acceleration(
      const int &xz, const int &iglob, const type_real &wxglll,
      const type_real &wzglll,
      const ScratchViewType<type_real> stress_integrand_1,
      const ScratchViewType<type_real> stress_integrand_2,
      const ScratchViewType<type_real> stress_integrand_3,
      const ScratchViewType<type_real> stress_integrand_4,
      const ScratchViewType<type_real> s_hprimewgll_xx,
      const ScratchViewType<type_real> s_hprimewgll_zz) const override;

private:
  int ispec;
  static specfem::compute::partial_derivatives partial_derivatives;
  static specfem::compute::properties properties;
  static specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
      field_dot_dot;
};

template <>
class element<specfem::enums::element::dimension::dim2,
              specfem::enums::element::medium::elastic,
              specfem::enums::element::quadrature::dynamic_quadrature_points,
              specfem::enums::element::property::isotropic>
    : public element<
          specfem::enums::element::dimension::dim2,
          specfem::enums::element::medium::elastic,
          specfem::enums::element::quadrature::dynamic_quadrature_points> {
public:
  using quadrature_points =
      specfem::enums::element::quadrature::dynamic_quadrature_points;

  template <typename T>
  using ScratchViewType =
      typename quadrature_points::template ScratchViewType<T>;

  KOKKOS_FUNCTION
  element() = default;

  KOKKOS_FUNCTION
  element(const int ispec,
          const specfem::compute::partial_derivatives partial_derivatives,
          const specfem::compute::properties properties,
          const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
              field_dot_dot);

  KOKKOS_FUNCTION void
  compute_gradient(const int &xz, const ScratchViewType<type_real> s_hprime_xx,
                   const ScratchViewType<type_real> s_hprime_zz,
                   const ScratchViewType<type_real> field_x,
                   const ScratchViewType<type_real> field_z, type_real &duxdxl,
                   type_real &duxdzl, type_real &duzdxl,
                   type_real &duzdzl) const override;

  KOKKOS_FUNCTION void
  compute_stress(const int &xz, const type_real &duxdxl,
                 const type_real &duxdzl, const type_real &duzdxl,
                 const type_real &duzdzl, type_real &stress_integrand_1l,
                 type_real &stress_integrand_2l, type_real &stress_integrand_3l,
                 type_real &stress_integrand_4l) const override;

  KOKKOS_FUNCTION void update_acceleration(
      const int &xz, const int &iglob, const type_real &wxglll,
      const type_real &wzglll,
      const ScratchViewType<type_real> stress_integrand_1,
      const ScratchViewType<type_real> stress_integrand_2,
      const ScratchViewType<type_real> stress_integrand_3,
      const ScratchViewType<type_real> stress_integrand_4,
      const ScratchViewType<type_real> s_hprimewgll_xx,
      const ScratchViewType<type_real> s_hprimewgll_zz) const override;

  KOKKOS_FUNCTION int get_ispec() const override { return this->ispec; }

private:
  int ispec;
  int ngllx;
  int ngllz;
  static specfem::compute::partial_derivatives partial_derivatives;
  static specfem::compute::properties properties;
  static specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
      field_dot_dot;
};
} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
