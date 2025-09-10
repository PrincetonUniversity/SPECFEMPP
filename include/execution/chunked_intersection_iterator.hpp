/**
 * @file chunked_intersection_iterator.hpp
 * @brief Chunked intersection iterator for processing edge-edge intersections
 *
 * This file provides iterators for processing intersections between mesh edges,
 * enabling efficient parallel computation of coupled boundary conditions and
 * interface operations in spectral element methods.
 *
 * @section Usage
 * @code{.cpp}
 * // Example: Process edge intersections with chunked iterator
 * #include "execution/chunked_intersection_iterator.hpp"
 * #include "execution/for_all.hpp"
 *
 * using ParallelConfig = specfem::parallel_config::default_chunk_edge_config<
 *     specfem::dimension::type::dim2, Kokkos::DefaultExecutionSpace>;
 *
 * // Create views for edge intersections
 * constexpr int num_intersections = 1000;
 * constexpr int num_points = 5;
 *
 * Kokkos::View<specfem::mesh_entity::edge*> self_edges("self_edges",
 * num_intersections); Kokkos::View<specfem::mesh_entity::edge*>
 * coupled_edges("coupled_edges", num_intersections);
 * Kokkos::View<int*[num_points]> self_storage("self_storage",
 * num_intersections); Kokkos::View<int*[num_points]>
 * coupled_storage("coupled_storage", num_intersections);
 *
 * // Initialize intersection edges
 * Kokkos::parallel_for("init_intersections", num_intersections,
 * KOKKOS_LAMBDA(int i) { self_edges(i) = specfem::mesh_entity::edge(i,
 * specfem::mesh_entity::type::top); coupled_edges(i) =
 * specfem::mesh_entity::edge(num_intersections-i-1,
 *                                                   specfem::mesh_entity::type::bottom);
 * });
 *
 * // Create chunked intersection iterator and process
 * specfem::execution::ChunkedIntersectionIterator iterator(
 *     ParallelConfig(), self_edges, coupled_edges, num_points);
 *
 * specfem::execution::for_all("process_intersections", iterator,
 *     KOKKOS_LAMBDA(const auto& index) {
 *         // Access coupled edge data
 *         auto self_idx = index.self_index;
 *         auto coupled_idx = index.coupled_index;
 *         int point = index.ipoint;
 *
 *         // Process intersection point
 *         Kokkos::atomic_add(&self_storage(self_idx.ispec, point), 1);
 *         Kokkos::atomic_add(&coupled_storage(coupled_idx.ispec, point), 1);
 *     });
 * @endcode
 */

#pragma once

#include "chunked_edge_iterator.hpp"
#include "macros.hpp"
#include "policy.hpp"
#include "specfem/point.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem::execution {

/**
 * @brief Index type representing a point on an edge-edge intersection interface
 *
 * This class encapsulates the coordinates and properties of a single point
 * located at the intersection of two mesh edges. It provides access to both the
 * self and coupled edge coordinates, enabling efficient processing of interface
 * conditions.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam KokkosIndexType Type of the underlying Kokkos policy index
 * @tparam ExecutionSpace Kokkos execution space for parallel operations
 *
 * @section Usage
 * @code{.cpp}
 * // Typically used within chunked intersection iterator lambdas
 * specfem::execution::for_all("process_intersections", iterator,
 *     KOKKOS_LAMBDA(const InterfacePointIndex<dim2, int, ExecutionSpace>&
 * index) {
 *         // Access interface point data
 *         auto interface_idx = index.get_index();
 *         auto self_point = interface_idx.self_index;    // Self edge
 * coordinates auto coupled_point = interface_idx.coupled_index; // Coupled edge
 * coordinates int point_pos = index.ipoint;                  // Position along
 * interface
 *
 *         // Process coupled boundary condition...
 *     });
 * @endcode
 */
template <specfem::dimension::type DimensionTag, typename KokkosIndexType,
          typename ExecutionSpace>
class InterfacePointIndex {
public:
  using index_type = specfem::point::interface_index<DimensionTag>;
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
   * @brief Get the interface index containing self and coupled coordinates
   *
   * @return const index_type& Reference to the interface index
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const index_type &get_index() const {
    return this->index; ///< Returns the local coordinates of the GLL point on
                        ///< the interface
  }

  /**
   * @brief Constructor for InterfacePointIndex
   *
   * @param self_index Local element coordinates of the self edge point
   * @param coupled_index Local element coordinates of the coupled edge point
   * @param ipoint Position of point along the interface (0 to num_points-1)
   * @param kokkos_index Underlying Kokkos policy index
   */
  KOKKOS_INLINE_FUNCTION
  InterfacePointIndex(
      const specfem::point::edge_index<DimensionTag> &self_index,
      const specfem::point::edge_index<DimensionTag> &coupled_index,
      const KokkosIndexType &kokkos_index)
      : index(self_index, coupled_index), kokkos_index(kokkos_index) {}

  /**
   * @brief Get iterator for this single interface point
   *
   * @return const iterator_type VoidIterator since this represents a single
   * point
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const iterator_type get_iterator() const { return iterator_type{}; }

public:
  int ipoint; ///< Position along interface edge (0 to num_points-1)

private:
  index_type index;             ///< Index of the GLL point on the interface
  KokkosIndexType kokkos_index; ///< Kokkos index type
};

/**
 * @brief Team-level iterator for processing intersection points within a chunk
 *
 * This iterator operates at the team level, processing intersection points
 * between two sets of edges (self and coupled). It combines two
 * ChunkEdgeIterators to provide synchronized access to corresponding points on
 * intersecting edges.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam ViewType Kokkos view type containing mesh edges
 * @tparam TeamMemberType Kokkos team member type
 *
 * @section IntersectionMapping
 * Each intersection point corresponds to:
 * - A point on the self edge (computed by self_iterator)
 * - A corresponding point on the coupled edge (computed by coupled_iterator)
 * - Both points share the same ipoint index along their respective edges
 *
 * @section Usage
 * @code{.cpp}
 * // Used internally by ChunkedIntersectionIterator
 * // Creates interface points from self and coupled edge iterators
 * @endcode
 */
template <specfem::dimension::type DimensionTag, typename ViewType,
          typename TeamMemberType>
class ChunkedEdgeIntersectionIterator
    : public TeamThreadRangePolicy<TeamMemberType, int> {
private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;

public:
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::TeamThreadRange
  using policy_index_type =
      typename base_type::policy_index_type; ///< Policy index type. Must be
                                             ///< convertible to integral type.
  using index_type = InterfacePointIndex<DimensionTag, policy_index_type,
                                         typename base_type::execution_space>;
  using execution_space =
      typename base_type::execution_space; ///< Execution space type.

  /**
   * @brief Convert linear index to interface point index
   *
   * Maps a linear index to interface point coordinates by computing
   * corresponding self and coupled edge points.
   *
   * @param i Linear index within the thread range
   * @return index_type InterfacePointIndex for the specified linear index
   */
  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &i) const {
    const auto self_index = self_iterator(i);
    const auto coupled_index = coupled_iterator(i);

    return index_type{ self_index.get_index(), coupled_index.get_index(),
                       self_index.get_policy_index() };
  }

  /**
   * @brief Constructor for team-level intersection iterator
   *
   * @param team_member Kokkos team member for parallel execution
   * @param self_edges View of self mesh edges
   * @param coupled_edges View of coupled mesh edges
   * @param num_points Number of GLL points per edge
   */
  KOKKOS_INLINE_FUNCTION
  ChunkedEdgeIntersectionIterator(const TeamMemberType &team_member,
                                  const ViewType &self_edges,
                                  const ViewType &coupled_edges,
                                  const int num_points)
      : base_type(team_member, self_edges.extent(0) * num_points),
        self_iterator(team_member, self_edges, num_points),
        coupled_iterator(team_member, coupled_edges, num_points) {}

  /**
   * @brief Get the self edge iterator
   *
   * @return const ChunkEdgeIterator& Reference to self edge iterator
   */
  KOKKOS_INLINE_FUNCTION
  const ChunkEdgeIterator<DimensionTag, ViewType, TeamMemberType> &
  get_self_iterator() const {
    return self_iterator;
  }

  /**
   * @brief Get the coupled edge iterator
   *
   * @return const ChunkEdgeIterator& Reference to coupled edge iterator
   */
  KOKKOS_INLINE_FUNCTION
  const ChunkEdgeIterator<DimensionTag, ViewType, TeamMemberType> &
  get_coupled_iterator() const {
    return coupled_iterator;
  }

private:
  ChunkEdgeIterator<DimensionTag, ViewType, TeamMemberType>
      self_iterator; ///< Iterator for self edges
  ChunkEdgeIterator<DimensionTag, ViewType, TeamMemberType>
      coupled_iterator; ///< Iterator for coupled edges
};

/**
 * @brief Chunk-level index for managing edge intersection processing within a
 * team
 *
 * This class serves as an intermediate index type that manages a chunk of edge
 * intersections assigned to a Kokkos team. It provides access to both self and
 * coupled edge chunks, their respective iterators, and the team-level
 * intersection iterator for processing interface points within the chunk.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam ViewType Kokkos view type containing the chunk of mesh edges
 * @tparam KokkosIndexType Type of the underlying Kokkos policy index (team
 * member)
 *
 * @section Architecture
 * ChunkEdgeIntersectionIndex acts as a bridge in the intersection iterator
 * hierarchy:
 * - High-level ChunkedIntersectionIterator (manages teams and intersection
 * chunks)
 * - Chunk-level ChunkEdgeIntersectionIndex (manages intersection chunk within a
 * team)
 * - Team-level ChunkedEdgeIntersectionIterator (processes intersection points
 * within chunk)
 * - Low-level InterfacePointIndex (represents individual interface points)
 *
 * @section DualEdgeManagement
 * Unlike ChunkEdgeIndex which manages a single edge set, this class manages:
 * - Self edges: Primary edge set (e.g., from domain A)
 * - Coupled edges: Secondary edge set (e.g., from domain B)
 * - Synchronized processing: Ensures corresponding points are processed
 * together
 *
 * @section Usage
 * @code{.cpp}
 * // Typically used internally within the chunked intersection iterator
 * hierarchy
 * // Not directly instantiated by user code
 *
 * // Example of accessing intersection chunk index in nested iteration:
 * specfem::execution::for_all("process_intersection_chunks", chunked_iterator,
 *     KOKKOS_LAMBDA(const auto& chunk_index) {
 *         // chunk_index is of type ChunkEdgeIntersectionIndex
 *         auto intersection_iterator = chunk_index.get_iterator();
 *         auto self_chunk = chunk_index.get_self_index();
 *         auto coupled_chunk = chunk_index.get_coupled_index();
 *         auto team_member = chunk_index.get_policy_index();
 *
 *         // Use intersection_iterator for processing interface points
 *         // within this chunk of intersections
 *     });
 * @endcode
 *
 * @section Applications
 * Common use cases for intersection chunk processing:
 * - Fluid-structure interaction at interfaces
 * - Acoustic-elastic coupling computations
 * - Domain decomposition boundary handling
 * - Multi-physics interface operations
 * - Periodic boundary condition enforcement
 */
template <specfem::dimension::type DimensionTag, typename ViewType,
          typename KokkosIndexType>
class ChunkEdgeIntersectionIndex {
private:
  using index_type = ChunkEdgeIntersectionIndex;

public:
  using iterator_type =
      ChunkedEdgeIntersectionIterator<DimensionTag, ViewType, KokkosIndexType>;

  /**
   * @brief Get the Kokkos policy index (team member) for this intersection
   * chunk
   *
   * @return const KokkosIndexType& Reference to the Kokkos team member
   *         that is responsible for processing this intersection chunk
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Get a reference to this intersection chunk index
   *
   * @return const index_type& Reference to this ChunkEdgeIntersectionIndex
   */
  KOKKOS_INLINE_FUNCTION
  const index_type &get_index() const { return *this; }

  /**
   * @brief Get the team-level iterator for processing intersection points in
   * this chunk
   *
   * @return const iterator_type& Reference to the
   * ChunkedEdgeIntersectionIterator that can be used to iterate over individual
   * interface points within this chunk
   */
  KOKKOS_INLINE_FUNCTION
  const iterator_type &get_iterator() const { return this->iterator; }

  /**
   * @brief Constructor for ChunkEdgeIntersectionIndex
   *
   * Creates a chunk-level intersection index managing both self and coupled
   * edge sets. Also initializes individual chunk indices for self and coupled
   * edges.
   *
   * @param self_edges View of self mesh edges for this specific chunk
   * @param coupled_edges View of coupled mesh edges for this specific chunk
   * @param num_points Number of GLL points per edge
   * @param kokkos_index Kokkos team member responsible for this intersection
   * chunk
   */
  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIntersectionIndex(const ViewType &self_edges,
                             const ViewType &coupled_edges,
                             const int num_points,
                             const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index),
        iterator(kokkos_index, self_edges, coupled_edges, num_points),
        self_index(self_edges, num_points, kokkos_index),
        coupled_index(coupled_edges, num_points, kokkos_index) {}

  /**
   * @brief Get the chunk index for self edges
   *
   * Provides access to the self edge chunk for independent processing
   * or when asymmetric operations are needed on the interface.
   *
   * @return const ChunkEdgeIndex& Reference to the self edge chunk index
   */
  KOKKOS_INLINE_FUNCTION
  const ChunkEdgeIndex<DimensionTag, ViewType, KokkosIndexType> &
  get_self_index() const {
    return self_index;
  }

  /**
   * @brief Get the chunk index for coupled edges
   *
   * Provides access to the coupled edge chunk for independent processing
   * or when asymmetric operations are needed on the interface.
   *
   * @return const ChunkEdgeIndex& Reference to the coupled edge chunk index
   */
  KOKKOS_INLINE_FUNCTION
  const ChunkEdgeIndex<DimensionTag, ViewType, KokkosIndexType> &
  get_coupled_index() const {
    return coupled_index;
  }

private:
  KokkosIndexType kokkos_index; ///< Kokkos team member for this intersection
                                ///< chunk
  iterator_type iterator; ///< Team-level intersection iterator for processing
                          ///< interface points
  ChunkEdgeIndex<DimensionTag, ViewType, KokkosIndexType>
      self_index; ///< Chunk index for self edges
  ChunkEdgeIndex<DimensionTag, ViewType, KokkosIndexType>
      coupled_index; ///< Chunk index for coupled edges
};

/**
 * @brief High-level chunked intersection iterator for efficient parallel
 * processing of edge intersections
 *
 * This is the main iterator class for processing intersections between mesh
 * edges in chunks. It manages two sets of edges (self and coupled) and
 * processes their intersections in parallel using Kokkos teams. This approach
 * is essential for coupled boundary conditions, interface operations, and
 * multi-domain computations in spectral element methods.
 *
 * @tparam ParallelConfig Configuration class defining execution parameters and
 * chunk size
 * @tparam ViewType Kokkos view type containing mesh edges
 *
 * @section Applications
 * The chunked intersection iterator is commonly used for:
 * - Fluid-structure interaction boundaries
 * - Acoustic-elastic interface coupling
 * - Domain decomposition interface handling
 * - Periodic boundary condition enforcement
 * - Multi-physics coupling operations
 *
 * @section Performance
 * The chunked approach provides benefits for intersection processing:
 * - Improved memory locality for coupled edge data
 * - Better load balancing across teams
 * - Reduced synchronization overhead
 * - Configurable chunk sizes for different architectures
 *
 * @section Usage
 * @code{.cpp}
 * #include "execution/chunked_intersection_iterator.hpp"
 * #include "execution/for_all.hpp"
 *
 * // Define parallel configuration
 * using ParallelConfig = specfem::parallel_config::default_chunk_edge_config<
 *     specfem::dimension::type::dim2, Kokkos::DefaultExecutionSpace>;
 *
 * // Create edge views for intersection
 * constexpr int num_intersections = 5000;
 * constexpr int num_points = 5;
 *
 * Kokkos::View<specfem::mesh_entity::edge*> self_edges("self_edges",
 * num_intersections); Kokkos::View<specfem::mesh_entity::edge*>
 * coupled_edges("coupled_edges", num_intersections);
 *
 * // Initialize intersection pairs
 * Kokkos::parallel_for("init_intersections", num_intersections,
 * KOKKOS_LAMBDA(int i) {
 *     // Self edges from domain A
 *     self_edges(i) = specfem::mesh_entity::edge(i,
 * specfem::mesh_entity::type::top);
 *
 *     // Coupled edges from domain B (often with different orientation)
 *     coupled_edges(i) = specfem::mesh_entity::edge(
 *         num_intersections - i - 1, specfem::mesh_entity::type::bottom);
 * });
 *
 * // Create storage for coupled calculations
 * Kokkos::View<double*[num_points]> self_field("self_field",
 * num_intersections); Kokkos::View<double*[num_points]>
 * coupled_field("coupled_field", num_intersections);
 * Kokkos::View<double*[num_points]> interface_flux("interface_flux",
 * num_intersections);
 *
 * // Create and use chunked intersection iterator
 * specfem::execution::ChunkedIntersectionIterator iterator(
 *     ParallelConfig(), self_edges, coupled_edges, num_points);
 *
 * specfem::execution::for_all("compute_interface_coupling", iterator,
 *     KOKKOS_LAMBDA(const auto& index) {
 *         // Access interface point data
 *         auto interface_idx = index.get_index();
 *         auto self_point = interface_idx.self_index;
 *         auto coupled_point = interface_idx.coupled_index;
 *         int point = index.ipoint;
 *
 *         // Compute interface coupling (e.g., acoustic-elastic)
 *         double self_value = self_field(self_point.ispec, point);
 *         double coupled_value = coupled_field(coupled_point.ispec, point);
 *
 *         // Apply coupling operator (simplified example)
 *         double flux = 0.5 * (self_value + coupled_value);
 *         interface_flux(self_point.ispec, point) = flux;
 *     });
 *
 * Kokkos::fence();
 * @endcode
 *
 * @section ChunkSize
 * The chunk size affects performance for intersection processing:
 * - GPU: 128-512 intersections per chunk (memory bandwidth limited)
 * - CPU: 32-128 intersections per chunk (cache hierarchy optimized)
 * - Consider memory access patterns of coupled data structures
 */
template <typename ParallelConfig, typename ViewType>
class ChunkedIntersectionIterator : public TeamPolicy<ParallelConfig> {
private:
  using base_type = TeamPolicy<ParallelConfig>; ///< Base policy type
  constexpr static auto simd_size = 1;
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
      ChunkEdgeIntersectionIndex<ParallelConfig::dimension,
                                 decltype(Kokkos::subview(
                                     std::declval<ViewType>(),
                                     std::declval<Kokkos::pair<int, int> >())),
                                 policy_index_type>;

  using execution_space =
      typename base_type::execution_space; ///< Execution space type.
  using base_index_type = specfem::point::interface_index<
      ParallelConfig::dimension>; ///< Index type to be used when calling @c
                                  ///< specfem::execution::for_all with this
                                  ///< iterator.

  /**
   * @brief Constructor with explicit edge views and point count
   *
   * @param self_edges View of self mesh edges (first side of intersection)
   * @param coupled_edges View of coupled mesh edges (second side of
   * intersection)
   * @param num_points Number of GLL points per edge
   */
  ChunkedIntersectionIterator(const ViewType self_edges,
                              const ViewType coupled_edges,
                              const int num_points)
      : self_edges(self_edges), coupled_edges(coupled_edges),
        num_points(num_points),
        base_type(((self_edges.extent(0) / chunk_size) +
                   ((self_edges.extent(0) % chunk_size) != 0)),
                  Kokkos::AUTO, Kokkos::AUTO) {}

  /**
   * @brief Constructor with parallel configuration
   *
   * @param config Parallel configuration (unused but required for interface
   * compatibility)
   * @param self_edges View of self mesh edges (first side of intersection)
   * @param coupled_edges View of coupled mesh edges (second side of
   * intersection)
   * @param num_points Number of GLL points per edge
   */
  ChunkedIntersectionIterator(const ParallelConfig, const ViewType self_edges,
                              const ViewType coupled_edges,
                              const int num_points)
      : ChunkedIntersectionIterator(self_edges, coupled_edges, num_points) {}

  /**
   * @brief Team operator for intersection chunk processing
   *
   * Creates a chunk-specific intersection index for the given team. Each team
   * processes a contiguous chunk of intersection pairs, improving memory
   * locality for coupled computations.
   *
   * @param team Kokkos team member
   * @return index_type Chunk intersection index for this team
   */
  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &team) const {
    const auto league_id = team.league_rank();
    const int start = league_id * chunk_size;
    const int end = ((start + chunk_size) > self_edges.extent(0))
                        ? self_edges.extent(0)
                        : (start + chunk_size);
    const auto my_self_edges =
        Kokkos::subview(self_edges, Kokkos::make_pair(start, end));
    const auto my_coupled_edges =
        Kokkos::subview(coupled_edges, Kokkos::make_pair(start, end));
    return index_type(my_self_edges, my_coupled_edges, num_points, team);
  }

  /**
   * @brief Set scratch memory size for teams
   *
   * Forwards scratch memory configuration to the underlying team policy.
   * Useful for teams that need temporary storage for intersection computations.
   *
   * @tparam Args Variadic template for scratch size arguments
   * @param args Arguments to forward to team policy
   * @return ChunkedIntersectionIterator& Reference to this iterator for
   * chaining
   */
  template <typename... Args>
  inline ChunkedIntersectionIterator &set_scratch_size(Args &&...args) {
    base_type::set_scratch_size(std::forward<Args>(args)...);
    return *this;
  }

private:
  ViewType self_edges;    ///< View of self edges (first side of intersections)
  ViewType coupled_edges; ///< View of coupled edges (second side of
                          ///< intersections)
  int num_points;         ///< Number of GLL points per edge
};

} // namespace specfem::execution
