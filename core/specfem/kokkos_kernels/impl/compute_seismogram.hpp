#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace kokkos_kernels {
namespace impl {
/**
 * @brief Compute the seismograms for the given medium.
 *
 * This function computes the seismograms for the specified medium type and
 * properties. It is specialized for different dimension tags, wavefield types,
 * medium tags, and property tags.
 *
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam WavefieldType Simulation wavefield type (e.g., forward, adjoint,
 * backward)
 * @tparam NGLL Number of GLL points
 * @tparam MediumTag Medium type (e.g., elastic, acoustic)
 * @tparam PropertyTag Material property type (e.g., isotropic, anisotropic)
 *
 * @param assembly SPECFEM++ assembly object.
 * @param isig_step Time step for which the seismograms are computed
 */
template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void compute_seismograms(specfem::assembly::assembly<DimensionTag> &assembly,
                         const int &isig_step);

} // namespace impl
} // namespace kokkos_kernels
} // namespace specfem
