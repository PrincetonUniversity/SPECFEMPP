#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "globals.h"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_dim3_compute_mass_matrix_elastic
 *
 */

/**
 * @ingroup specfem_medium_dim3_compute_mass_matrix_elastic
 * @brief Compute mass matrix inverse for 3D elastic isotropic media.
 *
 * **Mass matrix:**
 * \f$ M_{ij} = \rho \delta_{ij} \f$
 *
 * **Components:**
 * \f$ [M_x, M_y, M_z] = [\rho, \rho, \rho] \f$
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @tparam PropertyTag Property type (isotropic, anisotropic)
 * @param properties Material properties
 * @return Mass inverse components for \f$ [u_x, u_y, u_z] \f$
 */
template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION
    specfem::point::mass_inverse<specfem::dimension::type::dim3,
                                 specfem::element::medium_tag::elastic, UseSIMD>
    impl_mass_matrix_component(
        const specfem::point::properties<specfem::dimension::type::dim3,
                                         specfem::element::medium_tag::elastic,
                                         PropertyTag, UseSIMD> &properties);

} // namespace medium
} // namespace specfem
