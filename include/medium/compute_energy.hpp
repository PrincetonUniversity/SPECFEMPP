#pragma once

#include "compute/assembly/assembly.hpp"
#include "dim2/acoustic/isotropic/compute_energy.hpp"
#include "dim2/elastic/anisotropic/compute_energy.hpp"
#include "dim2/elastic/isotropic/compute_energy.hpp"
#include "dim2/poroelastic/isotropic/compute_energy.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <specfem::wavefield::simulation_field Wavefield,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename MemberType,
          typename IteratorType, typename QuadratureType,
          typename ChunkFieldType, typename CallbackFunctor>
KOKKOS_INLINE_FUNCTION void
compute_energy(const MemberType &team, const IteratorType &iterator,
               const specfem::compute::assembly &assembly,
               const QuadratureType &quadrature,
               const ChunkFieldType &element_field, CallbackFunctor callback) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto dimension_type = DimensionType;

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type, dimension_type>;
  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, medium_tag>;
  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, property_tag>;

  static_assert(ChunkFieldType::isChunkFieldType,
                "field is not a chunk field type");
  static_assert(
      ChunkFieldType::store_displacement && ChunkFieldType::store_velocity &&
          ChunkFieldType::store_acceleration,
      "field type needs to store displacement, velocity and acceleration");

  impl_compute_energy<Wavefield>(dimension_dispatch(), medium_dispatch(),
                                 property_dispatch(), team, iterator, assembly,
                                 quadrature, element_field, callback);

  return;
}
} // namespace medium
} // namespace specfem
