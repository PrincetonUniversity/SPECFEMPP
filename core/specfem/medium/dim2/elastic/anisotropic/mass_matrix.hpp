#pragma once

#include "globals.h"
#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium_physics {

// Using template specializations from isotropic case

// template <bool UseSIMD, specfem::element::property_tag PropertyTag>
// KOKKOS_FUNCTION specfem::point::field<specfem::element::dimension_tag::dim2,
//                                       specfem::element::medium_tag::elastic_psv,
//                                       false, false, false, true, UseSIMD>
// impl_mass_matrix_component(
//     const specfem::point::properties<specfem::element::dimension_tag::dim2,
//                                      specfem::element::medium_tag::elastic_psv,
//                                      PropertyTag, UseSIMD> &properties,
//     const specfem::point::jacobian_matrix<
//         specfem::element::dimension_tag::dim2, true, UseSIMD>
//         &jacobian_matrix);

} // namespace medium_physics
} // namespace specfem
