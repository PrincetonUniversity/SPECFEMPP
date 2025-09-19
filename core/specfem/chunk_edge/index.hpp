#pragma once

#include "enumerations/interface.hpp"
#include "execution/chunked_edge_iterator.hpp"
#include "specfem/data_access.hpp"

namespace specfem::chunk_edge {

template <specfem::dimension::type DimensionTag, typename ViewType,
          typename TeamMemberType>
class Index
    : public specfem::execution::ChunkEdgeIndex<DimensionTag, TeamMemberType,
                                                ViewType>,
      public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::index, DimensionTag, false> {
private:
  using base_type =
      specfem::execution::ChunkEdgeIndex<DimensionTag, TeamMemberType,
                                         ViewType>;

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

} // namespace specfem::chunk_edge
