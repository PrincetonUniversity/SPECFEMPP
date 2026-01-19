#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace kokkos_kernels {
namespace impl {
/**
 * @brief Compute the source interaction for the given medium.
 *
 * This function computes the source interaction for the specified medium type
 * and properties. It is specialized for different dimension tags, medium tags,
 * property tags, and boundary tags.
 *
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam WavefieldType Simulation wavefield type (e.g., forward, adjoint,
 * backward)
 * @tparam NGLL Number of GLL points
 * @tparam MediumTag Medium type (e.g., elastic, acoustic)
 * @tparam PropertyTag Material property type (e.g., isotropic, anisotropic)
 * @tparam BoundaryTag Boundary condition type (e.g., free_surface, absorbing)
 *
 * @param assembly SPECFEM++ assembly object.
 * @param timestep Time step for which the source interaction is computed
 */
template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
void compute_source_interaction(
    specfem::assembly::assembly<DimensionTag> &assembly, const int &timestep);
} // namespace impl
} // namespace kokkos_kernels
} // namespace specfem
