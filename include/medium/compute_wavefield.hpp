#pragma once

#include "dim2/acoustic/isotropic/wavefield.hpp"
#include "dim2/elastic/anisotropic/wavefield.hpp"
#include "dim2/elastic/isotropic/wavefield.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @brief Compute the values of wavefield of a given component within a spectral
 * element.
 *
 *
 * This function computes the wavefield values given the intrinsic field values
 * within that element. For example, for elastic medium  when the wavefield
 * component is pressure, the function computes the pressure values from the
 * displacement field values.
 *
 *
 * @ingroup MediumPhysics
 *
 * @tparam MediumTag The medium tag of the element
 * @tparam PropertyTag The property tag of the element
 * @tparam MemberType The kokkos team policy member type
 * @tparam IteratorType The iterator type specfem::iterator::chunk
 * @tparam ChunkFieldType Chunk field type that stores the intrinsic field
 * values specfem::chunk_element::field
 * @tparam QuadratureType The quadrature type that stores the lagrange
 * polynomial values specfem::element::quadrature
 * @tparam WavefieldViewType 4 dimensional Kokkos view (output)
 * @param team The kokkos team policy member
 * @param iterator The iterator to iterate over all the GLL points
 * @param assembly SPECFEM++ assembly object
 * @param quadrature The quadrature object containing lagrange polynomial values
 * @param field Instrinsic field values
 * @param wavefield_component The wavefield component to compute
 * @param wavefield_on_entire_grid The wavefield view to store the computed
 * values
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename MemberType,
          typename IteratorType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_INLINE_FUNCTION auto
compute_wavefield(const MemberType &team, const IteratorType &iterator,
                  const specfem::compute::assembly &assembly,
                  const QuadratureType &quadrature, const ChunkFieldType &field,
                  const specfem::wavefield::type &wavefield_component,
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

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type,
                             specfem::dimension::type::dim2>;
  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;
  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  impl_compute_wavefield(dimension_dispatch(), medium_dispatch(),
                         property_dispatch(), team, iterator, assembly,
                         quadrature, field, wavefield_component,
                         wavefield_on_entire_grid);

  return;
}

} // namespace medium
} // namespace specfem
