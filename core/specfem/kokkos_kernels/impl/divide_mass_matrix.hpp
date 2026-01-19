#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace kokkos_kernels {
namespace impl {

/**
 * @brief Compute the invese of the mass matrix for a given medium type.
 *
 * This function computes the inverse of the mass matrix for a given medium
 * type. It calls the specialized implementation for different medium types and
 * properties.
 *
 * @tparam DimensionTag Dimension of the problem
 * @tparam WavefieldType Type of the wavefield (e.g., elastic, acoustic)
 * @tparam MediumTag Medium tag (e.g., elastic, acoustic)
 * @tparam PropertyTag Property tag (e.g., isotropic, anisotropic)
 * @tparam BoundaryTag Boundary tag (e.g., none, stacey)
 *
 */
template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::element::medium_tag MediumTag>
void divide_mass_matrix(
    const specfem::assembly::assembly<DimensionTag> &assembly);
} // namespace impl

} // namespace kokkos_kernels
} // namespace specfem
