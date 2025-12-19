/**
 * @file chunked_edge_iterator.hpp
 * @brief Chunked edge iterator implementation for efficient parallel processing
 * of mesh edges
 *
 * This file provides iterators for processing mesh edges in chunks, enabling
 * efficient parallel execution on GPU and CPU architectures. The chunked
 * approach improves memory locality and load balancing in spectral element
 * computations.
 *
 * @section Usage
 * @code{.cpp}
 * // Example: Process edges with chunked iterator
 * #include "execution/chunked_edge_iterator.hpp"
 * #include "execution/for_all.hpp"
 *
 * using ParallelConfig =
 * specfem::parallel_configuration::default_chunk_edge_config<
 *     specfem::dimension::type::dim2, Kokkos::DefaultExecutionSpace>;
 *
 * // Create views for storage and edges
 * constexpr int num_points = 5;
 * Kokkos::View<int*[num_points], Kokkos::DefaultExecutionSpace>
 * storage("storage", num_edges);
 * Kokkos::View<specfem::mesh_entity::edge<specfem::dimension::type::dim2>*,
 * Kokkos::DefaultExecutionSpace> edges("edges", num_edges);
 *
 * // Initialize edges
 * Kokkos::parallel_for("init_edges", num_edges, KOKKOS_LAMBDA(int i) {
 *     edges(i) = specfem::mesh_entity::edge<specfem::dimension::type::dim2>(i,
 * specfem::mesh_entity::dim2::type::top);
 * });
 *
 * // Create chunked iterator and process edges
 * specfem::execution::ChunkedEdgeIterator iterator(ParallelConfig(), edges,
 * num_points); specfem::execution::for_all("process_edges", iterator,
 *     KOKKOS_LAMBDA(const auto& iterator_index) {
 *       const auto index = iterator_index.get_index();
 *         // Access edge point data
 *         int ispec = index.ispec;     // Element index
 *         int ipoint = index.ipoint;   // Point index along edge
 *
 *         // Perform computation on edge point
 *         Kokkos::atomic_add(&storage(ispec, ipoint), 1);
 *     });
 * @endcode
 */

#pragma once

#include "chunked_edge_iterator.hpp"
#include "policy.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem::chunk_edge {
// Forward declaration for EdgeIndex
template <specfem::dimension::type DimensionTag, typename ViewType,
          typename TeamMemberType>
class Index;
} // namespace specfem::chunk_edge

namespace specfem::execution {

// clang-format off
/**
 * @brief Index type representing a single point on a mesh edge
 *
 * This class encapsulates the coordinates and properties of a single
 * Gauss-Lobatto-Legendre (GLL) point located on a mesh edge. It provides access
 * to both the local element coordinates and the position along the edge.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam KokkosIndexType Type of the underlying Kokkos policy index
 * @tparam ExecutionSpace Kokkos execution space for parallel operations
 *
 * @section Usage
 * @code{.cpp}
 * // Typically used within chunked edge iterator lambdas
 * specfem::execution::for_all("process_edges", iterator,
 *     KOKKOS_LAMBDA(const auto & iterator_index) {
*        const auto index = iterator_index.get_index();
 *         int element_id = index.ispec;    // Element containing this edge point
 *         int z_coord = index.iz;          // Local z-coordinate within element
 *         int x_coord = index.ix;          // Local x-coordinate within element
 *         int point_pos = index.ipoint;    // Position along edge (0 to num_points-1)
 *
 *         // Process the edge point...
 *     });
 * @endcode
 */
// clang-format on
template <specfem::dimension::type DimensionTag, typename KokkosIndexType,
          typename ExecutionSpace>
class EdgePointIndex {
public:
  using index_type = specfem::point::edge_index<DimensionTag>;
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
  constexpr const KokkosIndexType get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Get a reference to this index
   *
   * @return const index_type& Reference to this EdgePointIndex
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const index_type &get_index() const { return this->index; }

  KOKKOS_INLINE_FUNCTION
  constexpr const index_type &get_local_index() const {
    return this->local_index; ///< Returns the local point index
  }

  /**
   * @brief Constructor for EdgePointIndex
   *
   * @param index Local element coordinates of the edge point
   * @param iedge Position of edge within the chunk
   * @param kokkos_index Underlying Kokkos policy index
   */
  KOKKOS_INLINE_FUNCTION
  EdgePointIndex(const specfem::point::edge_index<DimensionTag> &index_,
                 const int iedge, const KokkosIndexType &kokkos_index)
      : index(index_), local_index(index_.ispec, iedge, index_.ipoint,
                                   index_.iz, index_.ix, index_.edge_type),
        kokkos_index(kokkos_index) {}

  /**
   * @brief Get iterator for this single point
   *
   * @return const iterator_type VoidIterator since this represents a single
   * point
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const iterator_type get_iterator() const { return iterator_type{}; }

private:
  index_type index;       ///< Local element coordinates of the edge point
  index_type local_index; ///< Local element coordinates relative to chunk

  KokkosIndexType kokkos_index; ///< Kokkos index type
};

/**
 * @brief Team-level iterator for processing edge points within a chunk
 *
 * This iterator operates at the team level, distributing edge points among team
 * members for parallel processing. It computes local element coordinates for
 * each edge point based on the edge type and orientation.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam ViewType Kokkos view type containing mesh edges
 * @tparam TeamMemberType Kokkos team member type
 *
 * @section EdgeMapping
 * Edge points are mapped to element coordinates based on edge type:
 * - bottom: (iz=0, ix=ipoint)
 * - top: (iz=num_points-1, ix=num_points-1-ipoint)
 * - left: (iz=ipoint, ix=0)
 * - right: (iz=num_points-1-ipoint, ix=num_points-1)
 *
 * @section Usage
 * @code{.cpp}
 * // Used internally by ChunkedEdgeIterator, not typically instantiated
 * directly
 * // Access through the chunked iterator's operator()
 * @endcode
 */
template <specfem::dimension::type DimensionTag, typename ViewType,
          typename TeamMemberType>
class ChunkEdgeIterator : public TeamThreadRangePolicy<TeamMemberType, int> {
private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;

public:
  using base_policy_type = typename base_type::base_policy_type;
  using policy_index_type = typename base_type::policy_index_type;
  using index_type = EdgePointIndex<DimensionTag, policy_index_type,
                                    typename base_type::execution_space>;
  using execution_space = typename base_type::execution_space;

  /**
   * @brief Convert linear index to edge point index
   *
   * Maps a linear index to specific edge and point coordinates, handling
   * edge orientation and element mapping.
   *
   * @param i Linear index within the thread range
   * @return index_type EdgePointIndex for the specified linear index
   */
  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &i) const {
    const int iedge = i % nedges;
    const int ipoint = i / nedges;
    const auto edge = edges(iedge);

    return index_type{ edge(ipoint), iedge, i };
  }

  /**
   * @brief Constructor for team-level edge iterator
   *
   * @param team_member Kokkos team member for parallel execution
   * @param edges View of mesh edges to process
   * @param npoints Number of GLL points per edge
   */
  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIterator(const TeamMemberType &team_member, const ViewType &edges_)
      : num_points(edges_.n_points), nedges(edges_.n_edges),
        base_type(team_member, edges_.n_edges * edges_.n_points),
        edges(edges_) {}

  const int nedges; ///< Total number of edges in this chunk
private:
  ViewType edges;         ///< View of mesh edges to iterate over
  std::size_t num_points; ///< Number of GLL points per edge
};

/**
 * @brief Chunk-level index for managing edge processing within a team
 *
 * This class serves as an intermediate index type that manages a chunk of edges
 * assigned to a Kokkos team. It provides access to both the chunk-specific
 * index and the team-level iterator for processing edge points within the
 * chunk.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam ViewType Kokkos view type containing the chunk of mesh edges
 * @tparam KokkosIndexType Type of the underlying Kokkos policy index (team
 * member)
 *
 * @section Architecture
 * ChunkEdgeIndex acts as a bridge between:
 * - High-level ChunkedEdgeIterator (manages teams and chunks)
 * - Team-level ChunkEdgeIterator (processes edges within a chunk)
 * - Low-level EdgePointIndex (represents individual edge points)
 *
 * @section Usage
 * @code{.cpp}
 * // Typically used internally within the chunked iterator hierarchy
 * // Not directly instantiated by user code
 *
 * // Example of accessing chunk index in nested iteration:
 * specfem::execution::for_all("process_chunks", chunked_iterator,
 *     KOKKOS_LAMBDA(const auto& chunk_index) {
 *         // chunk_index is of type ChunkEdgeIndex
 *         auto team_iterator = chunk_index.get_iterator();
 *         auto policy_idx = chunk_index.get_policy_index();
 *
 *         // Use team_iterator for further nested iteration
 *         // within this chunk of edges
 *     });
 * @endcode
 *
 * @section Responsibilities
 * - Maintains reference to the chunk's edge subset view
 * - Stores the Kokkos team member for parallel execution
 * - Provides access to team-level iterator for edge processing
 * - Bridges between chunk-level and point-level operations
 */
template <specfem::dimension::type DimensionTag, typename ViewType,
          typename KokkosIndexType>
class ChunkEdgeIndex {
private:
  using index_type =
      specfem::chunk_edge::Index<DimensionTag, ViewType, KokkosIndexType>;

public:
  using iterator_type =
      ChunkEdgeIterator<DimensionTag, ViewType, KokkosIndexType>;

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
   *         metadata about this specific chunk of edges
   */
  KOKKOS_INLINE_FUNCTION
  const index_type get_index() const { return { *this }; }

  /**
   * @brief Get the team-level iterator for processing edges in this chunk
   *
   * @return const iterator_type& Reference to the ChunkEdgeIterator that
   *         can be used to iterate over individual edge points within this
   * chunk
   */
  KOKKOS_INLINE_FUNCTION
  const iterator_type &get_iterator() const { return this->iterator; }

  /**
   * @brief Constructor for ChunkEdgeIndex
   *
   * @param edges View of mesh edges for this specific chunk (subset of total
   * edges)
   * @param num_points Number of GLL points per edge
   * @param kokkos_index Kokkos team member responsible for this chunk
   */
  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndex(const ViewType &edges, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), iterator(kokkos_index, edges),
        edges(edges) {}

  KOKKOS_INLINE_FUNCTION int nedges() const { return iterator.nedges; }

  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<std::size_t, std::size_t> get_range() const {
    return Kokkos::make_pair(edges(0).edge_index,
                             edges(edges.n_edges - 1).edge_index + 1);
  }

private:
  KokkosIndexType kokkos_index; ///< Kokkos team member for this chunk
  iterator_type iterator; ///< Team-level iterator for edge processing within
                          ///< chunk
  ViewType edges;         ///< View of mesh edges in this chunk
};

/**
 * @brief High-level chunked edge iterator for efficient parallel edge
 * processing
 *
 * This is the main iterator class for processing mesh edges in chunks. It
 * divides the edge set into chunks of configurable size and processes each
 * chunk in parallel using Kokkos teams. This approach improves memory locality
 * and load balancing.
 *
 * @tparam ParallelConfig Configuration class defining execution parameters and
 * chunk size
 * @tparam ViewType Kokkos view type containing mesh edges
 *
 * @section Performance
 * The chunked approach provides several benefits:
 * - Improved memory locality by processing related edges together
 * - Better load balancing across teams
 * - Reduced memory pressure from large edge sets
 * - Configurable chunk sizes for different architectures
 *
 * @section Usage
 * @code{.cpp}
 * #include "execution/chunked_edge_iterator.hpp"
 * #include "execution/for_all.hpp"
 *
 * // Define parallel configuration
 * using ParallelConfig =
 * specfem::parallel_configuration::default_chunk_edge_config<
 *     specfem::dimension::type::dim2, Kokkos::DefaultExecutionSpace>;
 *
 * // Create edge view and initialize
 * constexpr int num_edges = 10000;
 * constexpr int num_points = 5;
 * Kokkos::View<specfem::mesh_entity::edge<specfem::dimension::type::dim2>*>
 * edges("edges", num_edges);
 *
 * // Initialize edges with proper element indices and types
 * Kokkos::parallel_for("init_edges", num_edges, KOKKOS_LAMBDA(int i) {
 *     edges(i) = specfem::mesh_entity::edge<specfem::dimension::type::dim2>(i,
 * specfem::mesh_entity::dim2::type::top);
 * });
 *
 * // Create storage for computation results
 * Kokkos::View<int*[num_points]> storage("storage", num_edges);
 *
 * // Create and use chunked iterator
 * specfem::execution::ChunkedEdgeIterator iterator(ParallelConfig(), edges,
 * num_points);
 *
 * specfem::execution::for_all("process_edges", iterator,
 *     KOKKOS_LAMBDA(const auto& index) {
 *         // Each thread processes one edge point
 *         int element_id = index.ispec;
 *         int point_id = index.ipoint;
 *
 *         // Perform computation on edge point
 *         Kokkos::atomic_add(&storage(element_id, point_id), 1);
 *     });
 *
 * Kokkos::fence();
 * @endcode
 *
 * @section ChunkSize
 * The chunk size is determined by ParallelConfig::chunk_size. Typical values:
 * - GPU: 32 edges per chunk
 * - CPU: 1 edge per chunk
 */
template <typename ParallelConfig, typename ViewType>
class ChunkedEdgeIterator : public TeamPolicy<ParallelConfig> {
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
      ChunkEdgeIndex<ParallelConfig::dimension, ViewType,
                     policy_index_type>; ///< Underlying index type. This index
                                         ///< will be passed to the closure when
                                         ///< calling @ref
                                         ///< specfem::execution::for_each_level
  using execution_space =
      typename base_type::execution_space; ///< Execution space type.
  using base_index_type = EdgePointIndex<
      ParallelConfig::dimension, int,
      typename base_type::execution_space>; ///< Index type
                                            ///< to be used
                                            ///< when calling
                                            ///< @ref
                                            ///< specfem::execution::for_all
                                            ///< with this
                                            ///< iterator.

  /**
   * @brief Constructor with explicit edge view and point count
   *
   * @param edges View of mesh edges to process
   */
  ChunkedEdgeIterator(const ViewType edges)
      : edges(edges), base_type(((edges.n_edges / chunk_size) +
                                 ((edges.n_edges % chunk_size) != 0)),
                                Kokkos::AUTO, Kokkos::AUTO) {}

  /**
   * @brief Constructor with parallel configuration
   *
   * @param config Parallel configuration (unused but required for interface
   * compatibility)
   * @param edges View of mesh edges to process
   */
  ChunkedEdgeIterator(const ParallelConfig, const ViewType edges)
      : ChunkedEdgeIterator(edges) {}

  /**
   * @brief Team operator for chunk processing
   *
   * Creates a chunk-specific index for the given team. Each team processes
   * a contiguous chunk of edges, improving memory locality.
   *
   * @param team Kokkos team member
   * @return index_type Chunk index for this team
   */
  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &team) const {
    const auto league_id = team.league_rank();
    const int start = league_id * chunk_size;
    const int end = (start + chunk_size < edges.n_edges) ? start + chunk_size
                                                         : edges.n_edges;
    return index_type{ edges(Kokkos::make_pair(start, end)), team };
  }

  /**
   * @brief Set scratch memory size for teams
   *
   * Forwards scratch memory configuration to the underlying team policy.
   *
   * @tparam Args Variadic template for scratch size arguments
   * @param args Arguments to forward to team policy
   * @return ChunkedEdgeIterator& Reference to this iterator for chaining
   */
  template <typename... Args>
  inline ChunkedEdgeIterator &set_scratch_size(Args &&...args) {
    base_type::set_scratch_size(std::forward<Args>(args)...);
    return *this;
  }

private:
  ViewType edges; ///< View of indices of edges within this iterator.
};

} // namespace specfem::execution

#include "specfem/chunk_edge/index.hpp"
