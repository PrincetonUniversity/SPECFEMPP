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
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 * @tparam MediumTag Medium type (acoustic, elastic, poroelastic)
 * @tparam PropertyTag Property type (isotropic, anisotropic, etc.)
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
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename ChunkIndexType,
          typename DisplacementFieldType, typename VelocityFieldType,
          typename AccelerationFieldType, typename QuadratureType,
          typename WavefieldViewType>
KOKKOS_INLINE_FUNCTION auto
compute_wavefield(const ChunkIndexType &chunk_index,
                  const specfem::assembly::assembly<DimensionTag> &assembly,
                  const QuadratureType &quadrature,
                  const DisplacementFieldType &displacement,
                  const VelocityFieldType &velocity,
                  const AccelerationFieldType &acceleration,
                  const specfem::enums::wavefield &wavefield_component,
                  WavefieldViewType wavefield_on_entire_grid) {

  static_assert((WavefieldViewType::rank() == 4 &&
                 DimensionTag == specfem::element::dimension_tag::dim2) ||
                    (WavefieldViewType::rank() == 5 &&
                     DimensionTag == specfem::element::dimension_tag::dim3),
                "wavefield_on_entire_grid needs to be a 4D for 2D view and 5D "
                "for 3D view");

  static_assert(DisplacementFieldType::medium_tag == MediumTag,
                "DisplacementFieldType medium tag does not match MediumTag");
  static_assert(VelocityFieldType::medium_tag == MediumTag,
                "VelocityFieldType medium tag does not match MediumTag");
  static_assert(AccelerationFieldType::medium_tag == MediumTag,
                "AccelerationFieldType medium tag does not match MediumTag");

  static_assert(DisplacementFieldType::dimension_tag == DimensionTag,
                "DisplacementFieldType dimension tag must match DimensionTag");
  static_assert(VelocityFieldType::dimension_tag == DimensionTag,
                "VelocityFieldType dimension tag must match DimensionTag");
  static_assert(AccelerationFieldType::dimension_tag == DimensionTag,
                "AccelerationFieldType dimension tag must match DimensionTag");

  static_assert(
      specfem::data_access::is_chunk_element<DisplacementFieldType>::value &&
          specfem::data_access::is_chunk_element<VelocityFieldType>::value &&
          specfem::data_access::is_chunk_element<AccelerationFieldType>::value,
      "All field types must be chunk view types");

  using dimension_dispatch =
      std::integral_constant<specfem::element::dimension_tag, DimensionTag>;
  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;
  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  impl_compute_wavefield(dimension_dispatch(), medium_dispatch(),
                         property_dispatch(), chunk_index, assembly, quadrature,
                         displacement, velocity, acceleration,
                         wavefield_component, wavefield_on_entire_grid);
  return;
} // compute_wavefield

} // namespace medium_physics
} // namespace specfem
