#pragma once

#include "enumerations/interface.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "specfem/data_access.hpp"

namespace specfem {
namespace chunk_element {

// clang-format off
/**
 * @brief Chunk element index for high-performance spectral element computations.
 *
 * This class provides a specialized indexing interface for accessing and iterating
 * over chunks of spectral elements. It combines the functionality of the chunked
 * domain iterator with the SPECFEM++ data access layer, enabling efficient
 * chunk-based processing of multiple elements simultaneously.
 *
 * The Index class is designed to work with team-based Kokkos execution patterns
 * where multiple elements are processed together in chunks to improve cache
 * locality, enable vectorization, and maximize computational throughput. It
 * provides both spatial indexing within elements (GLL points) and iteration
 * over element chunks.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3) of the elements
 * @tparam SIMD SIMD type configuration for vectorized operations
 * @tparam ViewType Kokkos view type for element indices storage
 * @tparam TeamMemberType Kokkos team member type for parallel execution
 *
 * @note This class inherits from both ChunkElementIndex (for iteration functionality)
 *       and Accessor (for data access layer integration), providing a complete
 *       interface for chunk-based element processing.
 *
 * @code{.cpp}
 * // Example: Using chunk index with field accessors
 * specfem::execution::ChunkedDomainIterator chunk(...);
 * specfem::execution::for_each_level("process_elements",
  *     chunk, KOKKOS_LAMBDA(const decltype(chunk)::index_type &index) {
  *         // Access the chunk element index
  *         auto chunk_index = index.get_index();
  *
  *         // Use chunk_index to access field data
  *         ChunkField field(chunk_index.team_scratch(0));
  *         specfem::assembly::load_on_device(chunk_index.get_index(), field_container, field);
  *         // Perform computations on the field
  *         ... (computation logic)
  *     });
 * @endcode
 *
 * @see specfem::execution::ChunkElementIndex for iteration functionality
 */
// clang-format on
template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType>
class Index
    : public specfem::execution::ChunkElementIndex<DimensionTag, SIMD, ViewType,
                                                   TeamMemberType>,
      public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_element,
          specfem::data_access::DataClassType::index, DimensionTag,
          SIMD::using_simd> {
private:
  /// @brief Base type providing chunk element iteration functionality
  using base_type =
      specfem::execution::ChunkElementIndex<DimensionTag, SIMD, ViewType,
                                            TeamMemberType>;

public:
  /// @brief Iterator type for traversing elements in the chunk
  using iterator_type = typename base_type::iterator_type;

  /**
   * @brief Construct index from existing chunk element index base.
   *
   * Creates a chunk element index by wrapping an existing ChunkElementIndex
   * object. This constructor is useful when you already have a base chunk
   * element index and want to add the data access layer functionality.
   *
   * @param base The base chunk element index to wrap
   *
   * @code{.cpp}
   * // Create base chunk element index
   * auto base_index = specfem::execution::ChunkElementIndex<...>(...);
   *
   * // Wrap it with data access layer
   * IndexType chunk_index(base_index);
   * @endcode
   */
  KOKKOS_INLINE_FUNCTION
  Index(const base_type &base) : base_type(base) {}

  KOKKOS_INLINE_FUNCTION
  Index(const ViewType indices, const int &ngllz, const int &ngllx,
        const TeamMemberType &kokkos_index)
      : base_type(indices, ngllz, ngllx, kokkos_index) {}
};

} // namespace chunk_element
} // namespace specfem
