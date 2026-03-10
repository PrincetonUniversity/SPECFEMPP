/**
 * @file chunked_intersection_iterator.hpp
 * @brief Chunked intersection iterator for processing interface intersections
 *
 * This file provides iterators for processing intersections between mesh
 * interfaces, enabling efficient parallel computation of coupled boundary
 * conditions and interface operations in spectral element methods.
 *
 * @section Usage
 * @code{.cpp}
 * // Example: Process intersections with chunked iterator
 * #include "execution/chunked_intersection_iterator.hpp"
 * #include "execution/for_all.hpp"
 *
 * using ParallelConfig =
 * specfem::parallel_configuration::default_chunk_edge_config<
 *     specfem::element::dimension_tag::dim2, Kokkos::DefaultExecutionSpace>;
 *
 * // Create views for intersections
 * constexpr int num_intersections = 1000;
 *
 * Kokkos::View<specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2>*>
 * self_intersections("self_intersections", num_intersections);
 * Kokkos::View<specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2>*>
 * coupled_intersections("coupled_intersections", num_intersections);
 *
 * // Initialize intersection pairs
 * Kokkos::parallel_for("init_intersections", num_intersections,
 * KOKKOS_LAMBDA(int i) { self_intersections(i) =
 * specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2>(i,
 * specfem::mesh_entity::dim2::type::top); coupled_intersections(i) =
 * specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2>(num_intersections-i-1,
 *                                                   specfem::mesh_entity::dim2::type::bottom);
 * });
 *
 * // Create chunked intersection iterator and process
 * specfem::execution::ChunkedIntersectionIterator iterator(
 *     ParallelConfig(), self_intersections, coupled_intersections);
 *
 * specfem::execution::for_all("process_intersections", iterator,
 *     KOKKOS_LAMBDA(const auto& index) {
 *         // Access coupled interface data
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
#include "chunked_face_iterator.hpp"
#include "policy.hpp"
#include "specfem/point.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem::execution {

/**
 * @brief Index type representing a point on an interface intersection
 *
 * This class encapsulates the coordinates and properties of a single point
 * located at the intersection of two mesh interfaces. It provides access to
 * both the self and coupled interface coordinates, enabling efficient
 * processing of interface conditions.
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
 *         auto self_point = interface_idx.self_index;    // Self
 * interface coordinates
 *         auto coupled_point = interface_idx.coupled_index; // Coupled
 * interface coordinates
 *         int point_pos = index.ipoint;                  // Position along
 * interface
 *
 *         // Process coupled boundary condition...
 *     });
 * @endcode
 */
template <specfem::element::dimension_tag DimensionTag,
          typename KokkosIndexType, typename ExecutionSpace>
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

  KOKKOS_INLINE_FUNCTION
  constexpr const index_type &get_local_index() const {
    return this->local_index; ///< Returns the local point index
  }

  /**
   * @brief Constructor for InterfacePointIndex
   *
   * @tparam MeshEntityType Type of mesh entity based on the dimension of the
   * interface (e.g., edge_index or face_index)
   *
   * @param self_index Local element coordinates of the self interface point
   * @param coupled_index Local element coordinates of the coupled interface
   * point
   * @param kokkos_index Underlying Kokkos policy index
   */
  template <typename MeshEntityType>
  KOKKOS_INLINE_FUNCTION
  InterfacePointIndex(const MeshEntityType &self_index,
                      const MeshEntityType &coupled_index,
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

private:
  index_type index;             ///< Index of the GLL point on the interface
  KokkosIndexType kokkos_index; ///< Kokkos index type
};

namespace impl {
template <specfem::element::dimension_tag DimensionTag>
struct InterfaceIteratorTypeSelector;

template <>
struct InterfaceIteratorTypeSelector<specfem::element::dimension_tag::dim2> {
  template <typename ViewType, typename KokkosIndexType>
  using type = specfem::execution::ChunkEdgeIterator<
      specfem::element::dimension_tag::dim2, ViewType, KokkosIndexType>;
};

template <>
struct InterfaceIteratorTypeSelector<specfem::element::dimension_tag::dim3> {
  template <typename ViewType, typename KokkosIndexType>
  using type = specfem::execution::ChunkFaceIterator<
      specfem::element::dimension_tag::dim3, ViewType, KokkosIndexType>;
};
} // namespace impl

/**
 * @brief Team-level iterator for processing intersection points within a chunk
 *
 * This iterator operates at the team level, processing intersection points
 * between two sets of mesh entities (self and coupled). It combines two
 * chunk iterators to provide synchronized access to corresponding points on
 * intersecting interfaces.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam ViewType Kokkos view type containing mesh intersections
 * @tparam TeamMemberType Kokkos team member type
 *
 * @section IntersectionMapping
 * Each intersection point corresponds to:
 * - A point on the self interface (computed by self_iterator)
 * - A corresponding point on the coupled interface (computed by
 * coupled_iterator)
 * - Both points share the same ipoint index along their respective interfaces
 *
 * @section Usage
 * @code{.cpp}
 * // Used internally by ChunkedIntersectionIterator
 * // Creates interface points from self and coupled interface iterators
 * @endcode
 */
template <specfem::element::dimension_tag DimensionTag, typename ViewType,
          typename TeamMemberType>
class ChunkedInterfaceIntersectionIterator
    : public TeamThreadRangePolicy<TeamMemberType, int> {
private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;
  using iterator_type = typename impl::InterfaceIteratorTypeSelector<
      DimensionTag>::template type<ViewType, TeamMemberType>;

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
   * corresponding self and coupled interface points.
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
   * @param self_intersections View of self mesh intersections
   * @param coupled_intersections View of coupled mesh intersections
   */
  KOKKOS_INLINE_FUNCTION
  ChunkedInterfaceIntersectionIterator(const TeamMemberType &team_member,
                                       const ViewType &self_intersections,
                                       const ViewType &coupled_intersections)
      : base_type(team_member, self_intersections.get_total_points()),
        self_iterator(team_member, self_intersections),
        coupled_iterator(team_member, coupled_intersections) {}

  /**
   * @brief Get the self interface iterator
   *
   * @return const iterator_type& Reference to self interface iterator
   */
  KOKKOS_INLINE_FUNCTION
  const iterator_type &get_self_iterator() const { return self_iterator; }

  /**
   * @brief Get the coupled interface iterator
   *
   * @return const iterator_type& Reference to coupled interface iterator
   */
  KOKKOS_INLINE_FUNCTION
  const iterator_type &get_coupled_iterator() const { return coupled_iterator; }

private:
  iterator_type self_iterator;    ///< Iterator for self intersections
  iterator_type coupled_iterator; ///< Iterator for coupled intersections
};

namespace impl {
template <specfem::element::dimension_tag DimensionTag>
struct ChunkInterfaceIndexTypeSelector;

template <>
struct ChunkInterfaceIndexTypeSelector<specfem::element::dimension_tag::dim2> {
  template <typename ViewType, typename KokkosIndexType>
  using type = ChunkEdgeIndex<specfem::element::dimension_tag::dim2, ViewType,
                              KokkosIndexType>;
};

template <>
struct ChunkInterfaceIndexTypeSelector<specfem::element::dimension_tag::dim3> {
  template <typename ViewType, typename KokkosIndexType>
  using type = ChunkFaceIndex<specfem::element::dimension_tag::dim3, ViewType,
                              KokkosIndexType>;
};
} // namespace impl

/**
 * @brief Chunk-level index for managing interface intersection processing
 * within a team
 *
 * This class serves as an intermediate index type that manages a chunk of
 * intersections assigned to a Kokkos team. It provides access to both self and
 * coupled interface chunks, their respective iterators, and the team-level
 * intersection iterator for processing interface points within the chunk.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam ViewType Kokkos view type containing the chunk of mesh intersections
 * @tparam KokkosIndexType Type of the underlying Kokkos policy index (team
 * member)
 *
 * @section Architecture
 * ChunkInterfaceIntersectionIndex acts as a bridge in the intersection
 * iterator hierarchy:
 * - High-level ChunkedIntersectionIterator (manages teams and intersection
 * chunks)
 * - Chunk-level ChunkInterfaceIntersectionIndex (manages intersection chunk
 * within a team)
 * - Team-level ChunkedInterfaceIntersectionIterator (processes intersection
 * points within chunk)
 * - Low-level InterfacePointIndex (represents individual interface points)
 *
 * @section DualEdgeManagement
 * Unlike ChunkEdgeIndex/ChunkFaceIndex which manages a single entity set,
 * this class manages:
 * - Self intersections: Primary intersection set (e.g., from domain A)
 * - Coupled intersections: Secondary intersection set (e.g., from domain B)
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
 *         // chunk_index is of type ChunkInterfaceIntersectionIndex;
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
template <specfem::element::dimension_tag DimensionTag, typename ViewType,
          typename KokkosIndexType>
class ChunkInterfaceIntersectionIndex {
private:
  using index_type = ChunkInterfaceIntersectionIndex;
  using interface_index_type = typename impl::ChunkInterfaceIndexTypeSelector<
      DimensionTag>::template type<ViewType, KokkosIndexType>;

public:
  using iterator_type =
      ChunkedInterfaceIntersectionIterator<DimensionTag, ViewType,
                                           KokkosIndexType>;

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
   * @return const index_type& Reference to this
   * ChunkInterfaceIntersectionIndex
   */
  KOKKOS_INLINE_FUNCTION
  const index_type &get_index() const { return *this; }

  /**
   * @brief Get the team-level iterator for processing intersection points in
   * this chunk
   *
   * @return const iterator_type& Reference to the
   * ChunkedInterfaceIntersectionIterator that can be used to iterate over
   * individual interface points within this chunk
   */
  KOKKOS_INLINE_FUNCTION
  const iterator_type &get_iterator() const { return this->iterator; }

  /**
   * @brief Constructor for ChunkInterfaceIntersectionIndex
   *
   * Creates a chunk-level intersection index managing both self and coupled
   * intersection sets. Also initializes individual chunk indices for self and
   * coupled intersections.
   *
   * @param self_intersections View of self mesh intersections for this specific
   * chunk
   * @param coupled_intersections View of coupled mesh intersections for this
   * specific chunk
   * @param kokkos_index Kokkos team member responsible for this intersection
   * chunk
   */
  KOKKOS_INLINE_FUNCTION
  ChunkInterfaceIntersectionIndex(const ViewType &self_intersections,
                                  const ViewType &coupled_intersections,
                                  const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index),
        iterator(kokkos_index, self_intersections, coupled_intersections),
        self_index(self_intersections, kokkos_index),
        coupled_index(coupled_intersections, kokkos_index) {}
  /**
   * @brief Get the chunk index for self intersections
   *
   * Provides access to the self intersection chunk for independent processing
   * or when asymmetric operations are needed on the interface.
   *
   * @return const interface_index_type& Reference to the self interface chunk
   * index
   */
  KOKKOS_INLINE_FUNCTION
  const interface_index_type &get_self_index() const { return self_index; }

  /**
   * @brief Get the chunk index for coupled intersections
   *
   * Provides access to the coupled intersection chunk for independent
   * processing or when asymmetric operations are needed on the interface.
   *
   * @return const interface_index_type& Reference to the coupled interface
   * chunk index
   */
  KOKKOS_INLINE_FUNCTION
  const interface_index_type &get_coupled_index() const {
    return coupled_index;
  }

private:
  KokkosIndexType kokkos_index; ///< Kokkos team member for this intersection
                                ///< chunk
  iterator_type iterator; ///< Team-level intersection iterator for processing
                          ///< interface points
  interface_index_type self_index;    ///< Chunk index for self intersections
  interface_index_type coupled_index; ///< Chunk index for coupled intersections
};

/**
 * @brief High-level chunked intersection iterator for efficient parallel
 * processing of interface intersections
 *
 * This is the main iterator class for processing intersections between mesh
 * interfaces in chunks. It manages two sets of interfaces (self and coupled)
 * and processes their intersections in parallel using Kokkos teams. This
 * approach is essential for coupled boundary conditions, interface operations,
 * and multi-domain computations in spectral element methods.
 *
 * @tparam ParallelConfig Configuration class defining execution parameters and
 * chunk size
 * @tparam ViewType Kokkos view type containing mesh intersections
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
 * - Improved memory locality for coupled interface data
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
 * using ParallelConfig =
 * specfem::parallel_configuration::default_chunk_edge_config<
 *     specfem::element::dimension_tag::dim2, Kokkos::DefaultExecutionSpace>;
 *
 * // Create intersection views
 * constexpr int num_intersections = 5000;
 *
 * Kokkos::View<specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2>*>
 * self_intersections("self_intersections", num_intersections);
 * Kokkos::View<specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2>*>
 * coupled_intersections("coupled_intersections", num_intersections);
 *
 * // Initialize intersection pairs
 * Kokkos::parallel_for("init_intersections", num_intersections,
 * KOKKOS_LAMBDA(int i) {
 *     // Self intersections from domain A
 *     self_intersections(i) =
 * specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2>(i,
 * specfem::mesh_entity::dim2::type::top);
 *
 *     // Coupled intersections from domain B (often with different orientation)
 *     coupled_intersections(i) =
 * specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2>(
 * num_intersections
 * - i - 1, specfem::mesh_entity::dim2::type::bottom);
 * });
 *
 * // Create and use chunked intersection iterator
 * specfem::execution::ChunkedIntersectionIterator iterator(
 *     ParallelConfig(), self_intersections, coupled_intersections);
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
      ChunkInterfaceIntersectionIndex<ParallelConfig::dimension, ViewType,
                                      policy_index_type>;

  using execution_space =
      typename base_type::execution_space; ///< Execution space type.
  using base_index_type = InterfacePointIndex<
      ParallelConfig::dimension, int,
      typename base_type::execution_space>; ///< Index type
                                            ///< to be used
                                            ///< when calling
                                            ///< @ref
                                            ///< specfem::execution::for_all
                                            ///< with this
                                            ///< iterator.

  /**
   * @brief Constructor with explicit intersection views
   *
   * @param self_intersections View of self mesh intersections (first side of
   * intersection)
   * @param coupled_intersections View of coupled mesh intersections (second
   * side of intersection)
   */
  ChunkedIntersectionIterator(const ViewType self_intersections,
                              const ViewType coupled_intersections)
      : self_intersections(self_intersections),
        coupled_intersections(coupled_intersections),
        base_type(((self_intersections.N / chunk_size) +
                   ((self_intersections.N % chunk_size) != 0)),
                  Kokkos::AUTO, Kokkos::AUTO) {}

  /**
   * @brief Constructor with parallel configuration
   *
   * @param config Parallel configuration (unused but required for interface
   * compatibility)
   * @param self_intersections View of self mesh intersections (first side of
   * intersection)
   * @param coupled_intersections View of coupled mesh intersections (second
   * side of intersection)
   */
  ChunkedIntersectionIterator(const ParallelConfig,
                              const ViewType self_intersections,
                              const ViewType coupled_intersections)
      : ChunkedIntersectionIterator(self_intersections, coupled_intersections) {
  }

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
    const int end = ((start + chunk_size) > self_intersections.N)
                        ? self_intersections.N
                        : (start + chunk_size);
    const auto my_self_intersections =
        self_intersections(Kokkos::make_pair(start, end));
    const auto my_coupled_intersections =
        coupled_intersections(Kokkos::make_pair(start, end));
    return index_type(my_self_intersections, my_coupled_intersections, team);
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
  ViewType self_intersections;    ///< View of self intersections (first side)
  ViewType coupled_intersections; ///< View of coupled intersections (second
                                  ///< side of intersections)
};

} // namespace specfem::execution
