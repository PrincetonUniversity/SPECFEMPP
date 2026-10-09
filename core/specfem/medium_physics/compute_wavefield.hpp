#pragma once

#include "specfem/medium/dim2/acoustic/isotropic/wavefield.hpp"
#include "specfem/medium/dim2/elastic/anisotropic/wavefield.hpp"
#include "specfem/medium/dim2/elastic/isotropic/wavefield.hpp"
#include "specfem/medium/dim2/elastic/isotropic_cosserat/wavefield.hpp"
#include "specfem/medium/dim2/poroelastic/isotropic/wavefield.hpp"
#include "specfem/medium/dim3/acoustic/isotropic/wavefield.hpp"
#include "specfem/medium/dim3/elastic/isotropic/wavefield.hpp"
#include "specfem/medium/dim3/elastic/isotropic_cosserat/wavefield.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @brief Convert intrinsic fields values to wavefield of interest on GLL grid.
 *
 * Computes specified wavefield component (displacement, velocity, acceleration,
 * pressure, stress, etc.) from intrinsic field values at GLL nodes for
 * different medium types using medium-specific implementations.
 *
 * @see specfem::compute::impl::compute_seismograms for usage example.
 *
 * @tparam Tags Tags type encoding dimension, medium, property, attenuation, and
 * SIMD tags (e.g., specfem::tags::Tags<dim2, elastic, isotropic,
 * attenuation_tag::none, false>)
 * @tparam ChunkIndexType Type of element chunk identifier
 * @tparam DisplacementFieldType Type of displacement field
 * @tparam VelocityFieldType Type of velocity field
 * @tparam AccelerationFieldType Type of acceleration field
 * @tparam QuadratureType Type of quadrature rule
 * @tparam WavefieldViewType Kokkos view type for output wavefield
 * @param chunk_index Element chunk identifier
 * @param assembly Spectral element assembly information
 * @param quadrature Quadrature rule for GLL nodes
 * @param displacement Displacement field at GLL nodes
 * @param velocity Velocity field at GLL nodes
 * @param acceleration Acceleration field at GLL nodes
 * @param wavefield_component Type of wavefield to compute
 * @param wavefield_on_entire_grid Output wavefield values on GLL grid
 */
template <typename Tags, typename ChunkIndexType,
          typename DisplacementFieldType, typename VelocityFieldType,
          typename AccelerationFieldType, typename QuadratureType,
          typename WavefieldViewType>
KOKKOS_INLINE_FUNCTION auto compute_wavefield(
    const ChunkIndexType &chunk_index,
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const QuadratureType &quadrature, const DisplacementFieldType &displacement,
    const VelocityFieldType &velocity,
    const AccelerationFieldType &acceleration,
    const specfem::enums::wavefield &wavefield_component,
    WavefieldViewType wavefield_on_entire_grid) {

  static_assert(
      (WavefieldViewType::rank() == 4 &&
       Tags::dimension_tag == specfem::element::dimension_tag::dim2) ||
          (WavefieldViewType::rank() == 5 &&
           Tags::dimension_tag == specfem::element::dimension_tag::dim3),
      "wavefield_on_entire_grid needs to be a 4D for 2D view and 5D "
      "for 3D view");

  impl_compute_wavefield<Tags>(chunk_index, assembly, quadrature, displacement,
                               velocity, acceleration, wavefield_component,
                               wavefield_on_entire_grid);
  return;
} // compute_wavefield

} // namespace medium_physics
} // namespace specfem
