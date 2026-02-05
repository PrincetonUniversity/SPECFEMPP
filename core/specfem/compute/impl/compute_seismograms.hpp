#pragma once

#include "enumerations/wavefield.hpp"
#include "specfem/assembly.hpp"
#include "specfem/element.hpp"

namespace specfem {
namespace compute {
namespace impl {
/**
 * @brief Compute the seismograms for the given medium.
 *
 * This function computes the seismograms for the specified medium type and
 * properties. It is specialized for different dimension tags, wavefield types,
 * medium tags, and property tags.
 *
 * @tparam WavefieldType Simulation wavefield type (e.g., forward, adjoint,
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * backward)
 * @tparam NGLL Number of GLL points
 * @tparam MediumTag Medium type (e.g., elastic, acoustic)
 * @tparam PropertyTag Material property type (e.g., isotropic, anisotropic)
 *
 * @param assembly SPECFEM++ assembly object.
 * @param isig_step Time step for which the seismograms are computed
 */
template <specfem::simulation::field_type WavefieldType,
          specfem::element::dimension_tag DimensionTag, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void compute_seismograms(specfem::assembly::assembly<DimensionTag> &assembly,
                         const int &isig_step);

} // namespace impl
} // namespace compute
} // namespace specfem
