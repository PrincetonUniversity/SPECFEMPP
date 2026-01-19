#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace kokkos_kernels {

namespace impl {
/**
 * @brief Compute material derivatives for the given medium.
 *
 * This function computes the material derivatives for the specified medium
 * type and properties. It is specialized for different dimension tags,
 * medium tags, and property tags.
 *
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam NGLL Number of GLL points
 * @tparam MediumTag Medium type (e.g., elastic, acoustic)
 * @tparam PropertyTag Material property type (e.g., isotropic, anisotropic)
 *
 * @param assembly SPECFEM++ assembly object.
 * @param dt Time step size (used for time-dependent computations)
 */
template <specfem::dimension::type DimensionTag, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void compute_material_derivatives(
    const specfem::assembly::assembly<DimensionTag> &assembly,
    const type_real &dt);
} // namespace impl

} // namespace kokkos_kernels
} // namespace specfem
