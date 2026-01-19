#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/execution.hpp"

namespace specfem::chunk_edge {

/**
 * @brief Edge index for chunked edge operations in spectral element
 * simulations.
 *
 * Combines execution context with data access patterns for processing chunks
 * of edges. Provides efficient indexing for edge-based computations
 * such as interface coupling and boundary conditions.
 *
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam ViewType Kokkos view type for edge index storage
 * @tparam TeamMemberType Kokkos team execution context
 */
template <specfem::dimension::type DimensionTag, typename ViewType,
          typename TeamMemberType>
class Index : public specfem::execution::ChunkEdgeIndex<DimensionTag, ViewType,
                                                        TeamMemberType>,
              public specfem::data_access::Accessor<
                  specfem::data_access::AccessorType::chunk_edge,
                  specfem::data_access::DataClassType::edge_index, DimensionTag,
                  false> {
private:
  /// @brief Base execution index type for chunk edge operations
  using base_type = specfem::execution::ChunkEdgeIndex<DimensionTag, ViewType,
                                                       TeamMemberType>;

public:
  /// @brief Iterator type for traversing elements in the chunk
  using iterator_type = typename base_type::iterator_type;

  /**
   * @brief Construct from existing chunk edge index base.
   *
   * @param base Base chunk edge index to wrap with data access layer
   */
  KOKKOS_INLINE_FUNCTION
  Index(const base_type &base) : base_type(base) {}

  /**
   * @brief Construct with explicit parameters.
   *
   * @param indices Edge indices view
   * @param ngllz Number of GLL points in z-dimension
   * @param ngllx Number of GLL points in x-dimension
   * @param kokkos_index Team member execution context
   */
  KOKKOS_INLINE_FUNCTION
  Index(const ViewType indices, const int &ngllz, const int &ngllx,
        const TeamMemberType &kokkos_index)
      : base_type(indices, ngllz, ngllx, kokkos_index) {}
};

} // namespace specfem::chunk_edge
