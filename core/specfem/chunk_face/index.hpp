#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/execution.hpp"

namespace specfem::chunk_face {

/**
 * @brief Face index for chunked face operations in spectral element
 * simulations.
 *
 * Combines execution context with data access patterns for processing chunks
 * of faces. Provides efficient indexing for face-based computations
 * such as interface coupling and boundary conditions in 3D.
 *
 * @tparam DimensionTag Spatial dimension (3D)
 * @tparam ViewType Kokkos view type for face index storage
 * @tparam TeamMemberType Kokkos team execution context
 */
template <specfem::element::dimension_tag DimensionTag, typename ViewType,
          typename TeamMemberType>
class Index : public specfem::execution::ChunkFaceIndex<DimensionTag, ViewType,
                                                        TeamMemberType>,
              public specfem::data_access::Accessor<
                  specfem::datatype::AccessorType::chunk_face,
                  specfem::data_access::DataClassType::face_index, DimensionTag,
                  false> {
private:
  /// @brief Base execution index type for chunk face operations
  using base_type = specfem::execution::ChunkFaceIndex<DimensionTag, ViewType,
                                                       TeamMemberType>;

public:
  /// @brief Iterator type for traversing elements in the chunk
  using iterator_type = typename base_type::iterator_type;

  /**
   * @brief Construct from existing chunk face index base.
   *
   * @param base Base chunk face index to wrap with data access layer
   */
  KOKKOS_INLINE_FUNCTION
  Index(const base_type &base) : base_type(base) {}

  /**
   * @brief Construct with explicit parameters.
   *
   * @param indices Face indices view
   * @param ngllz Number of GLL points in z-dimension
   * @param nglly Number of GLL points in y-dimension
   * @param ngllx Number of GLL points in x-dimension
   * @param kokkos_index Team member execution context
   */
  KOKKOS_INLINE_FUNCTION
  Index(const ViewType indices, const int &ngllz, const int &nglly,
        const int &ngllx, const TeamMemberType &kokkos_index)
      : base_type(indices, ngllz, nglly, ngllx, kokkos_index) {}
};

} // namespace specfem::chunk_face
