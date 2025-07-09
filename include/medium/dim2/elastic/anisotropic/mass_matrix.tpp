#pragma once

#include "mass_matrix.hpp"

// The commented instantiation is identical to the isotropic case and
// therefore not needed.

// template <bool UseSIMD, specfem::element::property_tag PropertyTag>
// KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
//                                       specfem::element::medium_tag::elastic,
//                                       false, false, false, true, UseSIMD>
// specfem::medium::impl_mass_matrix_component(
//     const specfem::point::properties<
//         specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
//         PropertyTag, UseSIMD> &properties,
//     const specfem::point::jacobian_matrix<
//         specfem::dimension::type::dim2, true, UseSIMD> &jacobian_matrix) {

//   if constexpr (specfem::globals::simulation_wave == specfem::wave::p_sv) {
//     return specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>(
//         jacobian_matrix.jacobian * properties.rho,
//         jacobian_matrix.jacobian * properties.rho);
//   } else if constexpr (specfem::globals::simulation_wave == specfem::wave::sh) {
//     return specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>(
//         jacobian_matrix.jacobian * properties.rho, 0);
//   } else {
//     static_assert("Unknown wave type");
//     return specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>(
//         0.0, 0.0); // dummy return
//   }
// }
