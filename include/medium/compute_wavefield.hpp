#pragma once

#include "dim2/acoustic/isotropic/wavefield.hpp"
#include "dim2/elastic/anisotropic/wavefield.hpp"
#include "dim2/elastic/isotropic/wavefield.hpp"
#include "dim2/elastic/isotropic_cosserat/wavefield.hpp"
#include "dim2/poroelastic/isotropic/wavefield.hpp"
#include "dim3/elastic/isotropic/wavefield.hpp"
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
 * @tparam ChunkIndexType Chunk index type that stores the indices and iterator
 * for elements in the chunk
 * @tparam ChunkFieldType Chunk field type that stores the intrinsic field
 * values specfem::chunk_element::field
 * @tparam QuadratureType The quadrature type that stores the lagrange
 * polynomial values specfem::element::quadrature
 * @tparam WavefieldViewType 4 dimensional Kokkos view (output)
 * @param chunk_index The chunk index that contains the spectral element indices
 * @param assembly SPECFEM++ assembly object
 * @param quadrature The quadrature object containing lagrange polynomial values
 * @param field Instrinsic field values
 * @param wavefield_component The wavefield component to compute
 * @param wavefield_on_entire_grid The wavefield view to store the computed
 * values
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename ChunkIndexType,
          typename DisplacementFieldType, typename VelocityFieldType,
          typename AccelerationFieldType, typename QuadratureType,
          typename WavefieldViewType>
KOKKOS_INLINE_FUNCTION auto compute_wavefield(
    const ChunkIndexType &chunk_index,
    const specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const QuadratureType &quadrature, const DisplacementFieldType &displacement,
    const VelocityFieldType &velocity,
    const AccelerationFieldType &acceleration,
    const specfem::wavefield::type &wavefield_component,
    WavefieldViewType wavefield_on_entire_grid) {

  static_assert(QuadratureType::store_hprime_gll,
                "quadrature type needs to store GLL points");
  static_assert(WavefieldViewType::rank() == 4,
                "wavefield_on_entire_grid needs to be a 4D view");

  static_assert(DisplacementFieldType::medium_tag == MediumTag,
                "DisplacementFieldType medium tag does not match MediumTag");
  static_assert(VelocityFieldType::medium_tag == MediumTag,
                "VelocityFieldType medium tag does not match MediumTag");
  static_assert(AccelerationFieldType::medium_tag == MediumTag,
                "AccelerationFieldType medium tag does not match MediumTag");

  static_assert(DisplacementFieldType::dimension_tag ==
                    specfem::dimension::type::dim2,
                "DisplacementFieldType dimension tag must be dim2");
  static_assert(VelocityFieldType::dimension_tag ==
                    specfem::dimension::type::dim2,
                "VelocityFieldType dimension tag must be dim2");
  static_assert(AccelerationFieldType::dimension_tag ==
                    specfem::dimension::type::dim2,
                "AccelerationFieldType dimension tag must be dim2");

  static_assert(
      specfem::data_access::is_chunk_element<DisplacementFieldType>::value &&
          specfem::data_access::is_chunk_element<VelocityFieldType>::value &&
          specfem::data_access::is_chunk_element<AccelerationFieldType>::value,
      "All field types must be chunk view types");

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type,
                             specfem::dimension::type::dim2>;
  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;
  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  impl_compute_wavefield(dimension_dispatch(), medium_dispatch(),
                         property_dispatch(), chunk_index, assembly, quadrature,
                         displacement, velocity, acceleration,
                         wavefield_component, wavefield_on_entire_grid);
  return;
} // git pull

} // namespace medium
} // namespace specfem
