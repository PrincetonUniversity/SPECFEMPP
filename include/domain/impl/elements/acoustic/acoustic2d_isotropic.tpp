#ifndef _DOMAIN_ACOUSTIC_ELEMENTS2D_ISOTROPIC_TPP
#define _DOMAIN_ACOUSTIC_ELEMENTS2D_ISOTROPIC_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/acoustic/acoustic2d_isotropic.hpp"
#include "domain/impl/elements/element.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <int N, typename T>
using StaticScratchViewType =
    typename specfem::enums::element::quadrature::static_quadrature_points<
        N>::template ScratchViewType<T>;

// using field_type = Kokkos::Subview<
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
//     std::remove_const_t<decltype(Kokkos::ALL)> >;

// -----------------------------------------------------------------------------
//                     SPECIALIZED ELEMENT
// -----------------------------------------------------------------------------
// template <int NGLL, typename BC>
// KOKKOS_FUNCTION specfem::domain::impl::elements::element<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic,
//     BC>::element(const specfem::compute::partial_derivatives
//                      &partial_derivatives,
//                  const specfem::compute::properties &properties,
//                  const specfem::compute::boundaries &boundary_conditions,
//                  const quadrature_points_type &quadrature_points) {

// #ifndef NDEBUG
//   assert(partial_derivatives.xix.extent(1) == NGLL);
//   assert(partial_derivatives.xix.extent(2) == NGLL);
//   assert(partial_derivatives.gammax.extent(1) == NGLL);
//   assert(partial_derivatives.gammax.extent(2) == NGLL);
//   assert(partial_derivatives.xiz.extent(1) == NGLL);
//   assert(partial_derivatives.xiz.extent(2) == NGLL);
//   assert(partial_derivatives.gammaz.extent(1) == NGLL);
//   assert(partial_derivatives.gammaz.extent(2) == NGLL);
//   assert(partial_derivatives.jacobian.extent(1) == NGLL);
//   assert(partial_derivatives.jacobian.extent(2) == NGLL);

//   // Properties
//   assert(properties.rho_inverse.extent(1) == NGLL);
//   assert(properties.rho_inverse.extent(2) == NGLL);
//   assert(properties.lambdaplus2mu_inverse.extent(1) == NGLL);
//   assert(properties.lambdaplus2mu_inverse.extent(2) == NGLL);
//   assert(properties.kappa.extent(1) == NGLL);
//   assert(properties.kappa.extent(2) == NGLL);
// #endif

//   this->xix = partial_derivatives.xix;
//   this->gammax = partial_derivatives.gammax;
//   this->xiz = partial_derivatives.xiz;
//   this->gammaz = partial_derivatives.gammaz;
//   this->jacobian = partial_derivatives.jacobian;
//   this->rho_inverse = properties.rho_inverse;
//   this->lambdaplus2mu_inverse = properties.lambdaplus2mu_inverse;
//   this->kappa = properties.kappa;

//   this->boundary_conditions =
//       boundary_conditions_type(boundary_conditions, quadrature_points);

//   return;
// }

template <int NGLL, typename BC>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic, BC>::
    compute_mass_matrix_component(
        const specfem::point::properties<medium_type::value,
                                         property_type::value> &properties,
        const specfem::point::partial_derivatives2 &partial_derivatives,
        specfem::kokkos::array_type<type_real, 1> &mass_matrix) const {

  constexpr int components = medium_type::components;

  static_assert(components == 1, "Acoustic medium has only one component");

  mass_matrix[0] = partial_derivatives.jacobian / properties.kappa;

  return;
}

// template <int NGLL, typename BC>
// template <specfem::enums::time_scheme::type time_scheme>
// KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic, BC>::
//     mass_time_contribution(
//         const int &ispec, const int &ielement, const int &xz,
//         const type_real &dt,
//         const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
//         specfem::kokkos::array_type<type_real, medium_type::components>
//             &rmass_inverse) const {

//   int ix, iz;
//   sub2ind(xz, NGLL, iz, ix);

//   const specfem::compute::element_partial_derivatives partial_derivatives(
//       this->xix(ispec, iz, ix), this->gammax(ispec, iz, ix),
//       this->xiz(ispec, iz, ix), this->gammaz(ispec, iz, ix),
//       this->jacobian(ispec, iz, ix));

//   const specfem::compute::element_properties<medium_type::value,
//                                              property_type::value>
//       properties(this->lambdaplus2mu_inverse(ispec, iz, ix),
//                  this->rho_inverse(ispec, iz, ix));

//   rmass_inverse[0] = 0.0;

//   // comppute mass matrix component
//   boundary_conditions.template mass_time_contribution<time_scheme>(
//       ielement, xz, dt, weight, partial_derivatives, properties,
//       rmass_inverse);

//   return;
// }

// template <int NGLL, typename BC>
// KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic, BC>::
//     compute_gradient(
//         const int &ispec, const int &ielement, const int &xz,
//         const ScratchViewType<type_real, 1> s_hprime_xx,
//         const ScratchViewType<type_real, 1> s_hprime_zz,
//         const ScratchViewType<type_real, medium_type::components> field_chi,
//         specfem::kokkos::array_type<type_real, 1> &dchidxl,
//         specfem::kokkos::array_type<type_real, 1> &dchidzl) const {

//   int ix, iz;
//   sub2ind(xz, NGLL, iz, ix);

//   const specfem::compute::element_partial_derivatives partial_derivatives =
//       specfem::compute::element_partial_derivatives(
//           this->xix(ispec, iz, ix), this->gammax(ispec, iz, ix),
//           this->xiz(ispec, iz, ix), this->gammaz(ispec, iz, ix));

//   type_real dchi_dxi = 0.0;
//   type_real dchi_dgamma = 0.0;

// #ifdef KOKKOS_ENABLE_CUDA
// #pragma unroll
// #endif
//   for (int l = 0; l < NGLL; l++) {
//     dchi_dxi += s_hprime_xx(ix, l, 0) * field_chi(iz, l, 0);
//     dchi_dgamma += s_hprime_zz(iz, l, 0) * field_chi(l, ix, 0);
//   }

//   // dchidx
//   dchidxl[0] = dchi_dxi * partial_derivatives.xix +
//                dchi_dgamma * partial_derivatives.gammax;

//   // dchidz
//   dchidzl[0] = dchi_dxi * partial_derivatives.xiz +
//                dchi_dgamma * partial_derivatives.gammaz;

//   boundary_conditions.enforce_gradient(ielement, xz, partial_derivatives,
//                                        dchidxl, dchidzl);

//   return;
// }

// template <int NGLL, typename BC>
// KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic, BC>::
//     compute_stress(
//         const int &ispec, const int &ielement, const int &xz,
//         const specfem::kokkos::array_type<type_real, 1> &dchidxl,
//         const specfem::kokkos::array_type<type_real, 1> &dchidzl,
//         specfem::kokkos::array_type<type_real, 1> &stress_integrand_xi,
//         specfem::kokkos::array_type<type_real, 1> &stress_integrand_gamma)
//         const {

//   int ix, iz;
//   sub2ind(xz, NGLL, iz, ix);

//   const specfem::compute::element_partial_derivatives partial_derivatives(
//       this->xix(ispec, iz, ix), this->gammax(ispec, iz, ix),
//       this->xiz(ispec, iz, ix), this->gammaz(ispec, iz, ix),
//       this->jacobian(ispec, iz, ix));

//   const specfem::compute::element_properties<medium_type::value,
//                                              property_type::value>
//       properties(this->lambdaplus2mu_inverse(ispec, iz, ix),
//                  this->rho_inverse(ispec, iz, ix));

//   // Precompute the factor
//   type_real fac = partial_derivatives.jacobian * properties.rho_inverse;

//   // Compute stress integrands 1 and 2
//   // Here it is extremely important that this seems at odds with
//   // equations (44) & (45) from Komatitsch and Tromp 2002 I. - Validation
//   // The equations are however missing dxi/dx, dxi/dz, dzeta/dx, dzeta/dz
//   // for the gradient of w^{\alpha\gamma}. In this->update_acceleration
//   // the weights for the integration and the interpolated values for the
//   // first derivatives of the lagrange polynomials are then collapsed
//   stress_integrand_xi[0] = fac * (partial_derivatives.xix * dchidxl[0] +
//                                   partial_derivatives.xiz * dchidzl[0]);
//   stress_integrand_gamma[0] = fac * (partial_derivatives.gammax * dchidxl[0]
//   +
//                                      partial_derivatives.gammaz *
//                                      dchidzl[0]);

//   boundary_conditions.enforce_stress(ielement, xz, partial_derivatives,
//                                      properties, stress_integrand_xi,
//                                      stress_integrand_gamma);

//   return;
// }

// template <int NGLL, typename BC>
// KOKKOS_INLINE_FUNCTION void specfem::domain::impl::elements::element<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic, BC>::
//     compute_acceleration(
//         const int &ispec, const int &ielement, const int &xz,
//         const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
//         const ScratchViewType<type_real, medium_type::components>
//             stress_integrand_xi,
//         const ScratchViewType<type_real, medium_type::components>
//             stress_integrand_gamma,
//         const ScratchViewType<type_real, medium_type::components>
//             s_hprimewgll_xx,
//         const ScratchViewType<type_real, medium_type::components>
//             s_hprimewgll_zz,
//         const specfem::kokkos::array_type<type_real, medium_type::components>
//             &velocity,
//         specfem::kokkos::array_type<type_real, medium_type::components>
//             &acceleration) const {

//   int ix, iz;
//   sub2ind(xz, NGLL, iz, ix);
//   type_real temp1l = 0.0;
//   type_real temp2l = 0.0;

//   constexpr int components = medium_type::components;

//   static_assert(components == 1, "Acoustic medium has only one component");

//   specfem::compute::element_partial_derivatives partial_derivatives;

//   specfem::compute::element_properties<medium_type::value,
//   property_type::value>
//       properties;

//   // populate partial derivatives only if the boundary is stacey
//   // or if the boundary is composite_stacey_dirichlet
//   if constexpr ((boundary_conditions_type::value ==
//                  specfem::enums::element::boundary_tag::stacey) ||
//                 (boundary_conditions_type::value ==
//                  specfem::enums::element::boundary_tag::
//                      composite_stacey_dirichlet)) {
//     partial_derivatives = specfem::compute::element_partial_derivatives(
//         this->xix(ispec, iz, ix), this->gammax(ispec, iz, ix),
//         this->xiz(ispec, iz, ix), this->gammaz(ispec, iz, ix),
//         this->jacobian(ispec, iz, ix));

//     properties = specfem::compute::element_properties<medium_type::value,
//                                                       property_type::value>(
//         this->lambdaplus2mu_inverse(ispec, iz, ix),
//         this->rho_inverse(ispec, iz, ix));
//   }

// #ifdef KOKKOS_ENABLE_CUDA
// #pragma unroll
// #endif
//   for (int l = 0; l < NGLL; l++) {
//     temp1l += s_hprimewgll_xx(ix, l, 0) * stress_integrand_xi(iz, l, 0);
//     temp2l += s_hprimewgll_zz(iz, l, 0) * stress_integrand_gamma(l, ix, 0);
//   }

//   acceleration[0] = -1.0 * ((weight[1] * temp1l) + (weight[0] * temp2l));

//   boundary_conditions.enforce_traction(ielement, xz, weight,
//                                        partial_derivatives, properties,
//                                        velocity, acceleration);

//   return;
// }

#endif
