#pragma once

#include "enumerations/interface.hpp"
#include "impl/field.hpp"
#include "specfem/data_access.hpp"

namespace specfem::chunk_edge {

/**
 * @brief Chunk edge displacement field accessor for high-performance
 * spectral edge computations.
 *
 * This class provides a specialized interface for accessing and manipulating
 * displacement field data across chunks of spectral edges. It inherits all
 * functionality from the base chunk edge field implementation while being
 * specifically typed for displacement data.
 *
 * The displacement class is optimized for processing multiple edges
 * simultaneously (chunk-based processing), which improves cache locality and
 * enables vectorization.
 *
 * @tparam ChunkSize Number of elements processed together in a chunk for
 * optimal performance
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points per spatial dimension
 * @tparam DimensionTag Spatial dimension (dim2 or dim3) of the displacement
 * field
 * @tparam MediumTag Medium type (acoustic, elastic, poroelastic, etc.)
 * @tparam UseSIMD Whether to enable SIMD vectorization for performance
 * optimization
 *
 * @see specfem::chunk_edge::velocity for velocity field accessor
 * @see specfem::chunk_edge::acceleration for acceleration field accessor
 */
template <int ChunkSize, int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
class displacement
    : public impl::field<ChunkSize, NGLL, DimensionTag, MediumTag,
                         specfem::data_access::DataClassType::displacement,
                         UseSIMD> {
private:
  /// @brief Type alias for the base chunk edge field implementation
  using base_type =
      impl::field<ChunkSize, NGLL, DimensionTag, MediumTag,
                  specfem::data_access::DataClassType::displacement, UseSIMD>;

public:
  /// @brief SIMD type for vectorized displacement operations across chunks
  using simd = typename base_type::simd;

  /// @brief Vector type for storing displacement data with chunk-optimized
  /// layout
  using value_type = typename base_type::value_type;

  /// @brief Inherit all constructors from the base chunk edge field
  /// implementation
  using base_type::base_type;
};

} // namespace specfem::chunk_edge
