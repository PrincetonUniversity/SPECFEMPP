#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/setup.hpp"

namespace specfem {
namespace compute {

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
template <int NGLL, typename Tags>
void compute_material_derivatives(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const type_real &dt);
} // namespace impl

} // namespace compute
} // namespace specfem
