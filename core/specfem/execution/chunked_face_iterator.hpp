/**
 * @file chunked_face_iterator.hpp
 * @brief Chunked face iterator implementation for efficient parallel processing
 * of mesh faces in 3D
 *
 * This file provides iterators for processing mesh faces in chunks, enabling
 * efficient parallel execution on GPU and CPU architectures. The chunked
 * approach improves memory locality and load balancing in 3D spectral element
 * computations.
 *
 * Faces are 2D surfaces in 3D elements with ngll x ngll quadrature points.
 */

#pragma once

#include "policy.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem::chunk_face {
// Forward declaration for FaceIndex
template <specfem::element::dimension_tag DimensionTag, typename ViewType,
          typename TeamMemberType>
class Index;
} // namespace specfem::chunk_face

namespace specfem::execution {

/**
 * @brief Index type representing a single point on a mesh face
 *
 * This class encapsulates the coordinates and properties of a single
 * Gauss-Lobatto-Legendre (GLL) point located on a mesh face. It provides access
 * to both the local element coordinates and the position on the face.
 *
 * @tparam DimensionTag Spatial dimension (3D)
 * @tparam KokkosIndexType Type of the underlying Kokkos policy index
 * @tparam ExecutionSpace Kokkos execution space for parallel operations
 */
template <specfem::element::dimension_tag DimensionTag,
          typename KokkosIndexType, typename ExecutionSpace>
class FacePointIndex {
public:
  using index_type = specfem::point::face_index<DimensionTag>;
  using iterator_type =
      VoidIterator<ExecutionSpace>; ///< Iterator type used to iterate over
                                    ///< GLL points within this index.
                                    ///< @c VoidIterator is used when the
                                    ///< index refers to a single GLL point.

  /**
   * @brief Get the policy index that defined this point index.
   *
   * @return const KokkosIndexType The policy index that defined this point
   * index.
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Get a reference to this index
   *
   * @return const index_type& Reference to this FacePointIndex
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const index_type &get_index() const { return this->index; }

  KOKKOS_INLINE_FUNCTION
  constexpr const index_type &get_local_index() const {
    return this->local_index; ///< Returns the local point index
  }

  /**
   * @brief Constructor for FacePointIndex
   *
   * @param index Local element coordinates of the face point
   * @param iface Position of face within the chunk
   * @param kokkos_index Underlying Kokkos policy index
   */
  KOKKOS_INLINE_FUNCTION
  FacePointIndex(const specfem::point::face_index<DimensionTag> &index_,
                 const int iface, const KokkosIndexType &kokkos_index)
      : index(index_),
        local_index(index_.ispec, iface, index_.ipoint_i, index_.ipoint_j,
                    index_.iz, index_.iy, index_.ix, index_.face_type),
        kokkos_index(kokkos_index) {}

  /**
   * @brief Get iterator for this single point
   *
   * @return const iterator_type VoidIterator since this represents a single
   * point
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const iterator_type &get_iterator() const {
    return iterator_type{};
  }

  /// @brief Tail-skip flag required by the for_each_level dispatcher.
  /// Face points have no tail-skip semantics, so this is always false.
  KOKKOS_INLINE_FUNCTION
  constexpr bool is_end() const { return false; }

private:
  index_type index;       ///< Local element coordinates of the face point
  index_type local_index; ///< Local element coordinates relative to chunk

  KokkosIndexType kokkos_index; ///< Kokkos index type
};

/**
 * @brief Team-level iterator for processing face points within a chunk
 *
 * This iterator operates at the team level, distributing face points among team
 * members for parallel processing. It computes local element coordinates for
 * each face point based on the face type and orientation.
 *
 * @tparam DimensionTag Spatial dimension (3D)
 * @tparam ViewType Kokkos view type containing mesh faces
 * @tparam TeamMemberType Kokkos team member type
 */
template <specfem::element::dimension_tag DimensionTag, typename ViewType,
          typename TeamMemberType>
class ChunkFaceIterator : public TeamThreadRangePolicy<TeamMemberType, int> {
private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;

public:
  using base_policy_type = typename base_type::base_policy_type;
  using policy_index_type = typename base_type::policy_index_type;
  using index_type = FacePointIndex<DimensionTag, policy_index_type,
                                    typename base_type::execution_space>;
  using execution_space = typename base_type::execution_space;

  /**
   * @brief Convert linear index to face point index
   *
   * Maps a linear index to specific face and point coordinates, handling
   * face orientation and element mapping.
   *
   * @param i Linear index within the thread range
   * @return index_type FacePointIndex for the specified linear index
   */
  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &i) const {
    const int points_per_face = num_points * num_points;
    const int iface = i / points_per_face;
    const int point_idx = i % points_per_face;
    const int ipoint_i = point_idx / num_points;
    const int ipoint_j = point_idx % num_points;

    // Get the face point index from the faces view
    // The ViewType must provide operator()(iface)(ipoint_i, ipoint_j)
    return index_type{ faces(iface)(ipoint_i, ipoint_j), iface, i };
  }

  /**
   * @brief Constructor for team-level face iterator
   *
   * @param team_member Kokkos team member for parallel execution
   * @param faces View of mesh faces to process
   */
  KOKKOS_INLINE_FUNCTION
  ChunkFaceIterator(const TeamMemberType &team_member, const ViewType &faces_)
      : num_points(faces_.n_points), nfaces(faces_.N),
        base_type(team_member, faces_.N * faces_.n_points * faces_.n_points),
        faces(faces_) {}

  const int nfaces; ///< Total number of faces in this chunk
private:
  ViewType faces;         ///< View of mesh faces to iterate over
  std::size_t num_points; ///< Number of GLL points per face dimension
};

/**
 * @brief Chunk-level index for managing face processing within a team
 *
 * This class serves as an intermediate index type that manages a chunk of faces
 * assigned to a Kokkos team. It provides access to both the chunk-specific
 * index and the team-level iterator for processing face points within the
 * chunk.
 *
 * @tparam DimensionTag Spatial dimension (3D)
 * @tparam ViewType Kokkos view type containing the chunk of mesh faces
 * @tparam KokkosIndexType Type of the underlying Kokkos policy index (team
 * member)
 */
template <specfem::element::dimension_tag DimensionTag, typename ViewType,
          typename KokkosIndexType>
class ChunkFaceIndex {
private:
  using index_type =
      specfem::chunk_face::Index<DimensionTag, ViewType, KokkosIndexType>;

public:
  using iterator_type =
      ChunkFaceIterator<DimensionTag, ViewType, KokkosIndexType>;

  /**
   * @brief Get the Kokkos policy index (team member) for this chunk
   *
   * @return const KokkosIndexType& Reference to the Kokkos team member
   *         that is responsible for processing this chunk
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Get the chunk-specific index
   *
   * @return const index_type& Reference to the chunk index containing
   *         metadata about this specific chunk of faces
   */
  KOKKOS_INLINE_FUNCTION
  const index_type get_index() const { return { *this }; }

  /**
   * @brief Get the team-level iterator for processing faces in this chunk
   *
   * @return const iterator_type& Reference to the ChunkFaceIterator that
   *         can be used to iterate over individual face points within this
   * chunk
   */
  KOKKOS_INLINE_FUNCTION
  const iterator_type &get_iterator() const { return this->iterator; }

  /**
   * @brief Constructor for ChunkFaceIndex
   *
   * @param faces View of mesh faces for this specific chunk (subset of total
   * faces)
   * @param kokkos_index Kokkos team member responsible for this chunk
   */
  KOKKOS_INLINE_FUNCTION
  ChunkFaceIndex(const ViewType &faces, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), iterator(kokkos_index, faces),
        faces(faces) {}

  KOKKOS_INLINE_FUNCTION int nfaces() const { return iterator.nfaces; }

  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<std::size_t, std::size_t> get_range() const {
    return Kokkos::make_pair(faces(0).face_index,
                             faces(faces.N - 1).face_index + 1);
  }

private:
  KokkosIndexType kokkos_index; ///< Kokkos team member for this chunk
  iterator_type iterator; ///< Team-level iterator for face processing within
                          ///< chunk
  ViewType faces;         ///< View of mesh faces in this chunk
};

/**
 * @brief High-level chunked face iterator for efficient parallel face
 * processing in 3D
 *
 * This is the main iterator class for processing mesh faces in chunks. It
 * divides the face set into chunks of configurable size and processes each
 * chunk in parallel using Kokkos teams. This approach improves memory locality
 * and load balancing.
 *
 * @tparam ParallelConfig Configuration class defining execution parameters and
 * chunk size
 * @tparam ViewType Kokkos view type containing mesh faces
 */
template <typename ParallelConfig, typename ViewType>
class ChunkedFaceIterator : public TeamPolicy<ParallelConfig> {
private:
  using base_type = TeamPolicy<ParallelConfig>;
  constexpr static auto simd_size = 1;
  constexpr static auto using_simd = false;
  constexpr static auto chunk_size = ParallelConfig::chunk_size; ///< Chunk size

public:
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::TeamPolicy`
  using policy_index_type = typename base_type::
      policy_index_type; ///< Policy index type.
                         ///< Evaluates to
                         ///< @c Kokkos::TeamPolicy::member_type
  using index_type =
      ChunkFaceIndex<ParallelConfig::dimension, ViewType,
                     policy_index_type>; ///< Underlying index type. This index
                                         ///< will be passed to the closure when
                                         ///< calling @ref
                                         ///< specfem::execution::for_each_level
  using execution_space =
      typename base_type::execution_space; ///< Execution space type.
  using base_index_type = FacePointIndex<
      ParallelConfig::dimension, int,
      typename base_type::execution_space>; ///< Index type
                                            ///< to be used
                                            ///< when calling
                                            ///< @ref
                                            ///< specfem::execution::for_all
                                            ///< with this
                                            ///< iterator.

  /**
   * @brief Constructor with explicit face view
   *
   * @param faces View of mesh faces to process
   */
  ChunkedFaceIterator(const ViewType faces)
      : faces(faces),
        base_type(((faces.N / chunk_size) + ((faces.N % chunk_size) != 0)),
                  Kokkos::AUTO, Kokkos::AUTO) {}

  /**
   * @brief Constructor with parallel configuration
   *
   * @param config Parallel configuration (unused but required for interface
   * compatibility)
   * @param faces View of mesh faces to process
   */
  ChunkedFaceIterator(const ParallelConfig, const ViewType faces)
      : ChunkedFaceIterator(faces) {}

  /**
   * @brief Team operator for chunk processing
   *
   * Creates a chunk-specific index for the given team. Each team processes
   * a contiguous chunk of faces, improving memory locality.
   *
   * @param team Kokkos team member
   * @return index_type Chunk index for this team
   */
  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &team) const {
    const auto league_id = team.league_rank();
    const int start = league_id * chunk_size;
    const int end =
        (start + chunk_size < faces.N) ? start + chunk_size : faces.N;
    return index_type{ faces(Kokkos::make_pair(start, end)), team };
  }

  /**
   * @brief Set scratch memory size for teams
   *
   * Forwards scratch memory configuration to the underlying team policy.
   *
   * @tparam Args Variadic template for scratch size arguments
   * @param args Arguments to forward to team policy
   * @return ChunkedFaceIterator& Reference to this iterator for chaining
   */
  template <typename... Args>
  inline ChunkedFaceIterator &set_scratch_size(Args &&...args) {
    base_type::set_scratch_size(std::forward<Args>(args)...);
    return *this;
  }

private:
  ViewType faces; ///< View of indices of faces within this iterator.
};

} // namespace specfem::execution

#include "specfem/chunk_face/index.hpp"
