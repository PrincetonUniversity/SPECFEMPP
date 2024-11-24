#pragma once

#include "acoustic_isotropic2d/acoustic_isotropic2d.hpp"
#include "elastic_isotropic2d/elastic_isotropic2d.hpp"

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename MemberType,
          typename IteratorType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_INLINE_FUNCTION void
compute_wavefield(const MemberType &team, const IteratorType &iterator,
                  const specfem::compute::assembly &assembly,
                  const QuadratureType &quadrature, const ChunkFieldType &field,
                  const specfem::wavefield::component &wavefield_component,
                  WavefieldViewType wavefield_on_entire_grid) {

  static_assert(ChunkFieldType::isChunkFieldType,
                "field is not a chunk field type");
  static_assert(
      ChunkFieldType::store_displacement && ChunkFieldType::store_velocity &&
          ChunkFieldType::store_acceleration,
      "field type needs to store displacement, velocity and acceleration");
  static_assert(QuadratureType::store_hprime_gll,
                "quadrature type needs to store GLL points");
  static_assert(WavefieldViewType::rank() == 4,
                "wavefield_on_entire_grid needs to be a 4D view");

  static_assert(ChunkFieldType::medium_tag == MediumTag,
                "field type needs to have the same medium tag as the function");

  impl_compute_wavefield<MediumTag, PropertyTag>(
      team, iterator, assembly, quadrature, field, wavefield_component,
      wavefield_on_entire_grid);
}

} // namespace medium
} // namespace specfem
