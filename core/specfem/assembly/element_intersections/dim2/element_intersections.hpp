#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/element_coupling/flux_scheme_configuration.hpp"
#include "specfem/enums.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Individual 2D edge representation with quadrature point access
 *
 * This structure represents a single edge in the 2D spectral element mesh,
 * providing access to quadrature points along the edge for coupling
 * computations, boundary condition enforcement, and flux calculations.
 *
 * @tparam ExecutionSpace Kokkos execution space (host or device)
 */
template <typename ExecutionSpace> struct Edge {
  int n_points; ///< Number of quadrature points on this edge
  using IndexView = Kokkos::View<int *, Kokkos::LayoutStride,
                                 ExecutionSpace>; ///< View
                                                  ///< type for
                                                  ///< quadrature
                                                  ///< indices
  int element_index; ///< Index of the spectral element containing this edge
  int edge_index;    ///< Local edge index within the element
  specfem::mesh_entity::dim2::type edge_type; ///< 2D edge type (boundary
                                              ///< classification)
  IndexView iz; ///< Quadrature point indices in z-direction
  IndexView ix; ///< Quadrature point indices in x-direction

  /**
   * @brief Construct edge with quadrature point information.
   *
   * @param n_points Number of quadrature points on the edge
   * @param element_index Element containing this edge
   * @param edge_index Local edge index within element
   * @param edge_type 2D edge classification type
   * @param iz Z-direction quadrature indices
   * @param ix X-direction quadrature indices
   */
  KOKKOS_INLINE_FUNCTION
  Edge(const int n_points, const int element_index, const int edge_index,
       const specfem::mesh_entity::dim2::type edge_type, const IndexView iz,
       const IndexView ix)
      : n_points(n_points), element_index(element_index),
        edge_index(edge_index), edge_type(edge_type), iz(iz), ix(ix) {}

  /**
   * @brief Access quadrature point on the edge.
   *
   * @param point_id Quadrature point index along the edge
   * @return 2D edge index structure for the specified quadrature point
   */
  KOKKOS_INLINE_FUNCTION
  specfem::point::edge_index<specfem::element::dimension_tag::dim2>
  operator()(const int point_id) const {
    return { element_index, edge_index,   point_id,
             iz(point_id),  ix(point_id), edge_type };
  }
};

/**
 * @brief Collection of 2D edges with parallel access capabilities
 *
 * This structure manages collections of edges for efficient parallel
 * processing of edge-based operations such as coupling computations,
 * boundary condition enforcement, and flux calculations in 2D.
 *
 * @tparam ExecutionSpace Kokkos execution space (host or device)
 * @tparam Layout Memory layout for Kokkos views
 */
template <typename ExecutionSpace,
          typename Layout = typename ExecutionSpace::array_layout>
struct EdgeView {
  int N;        ///< Number of intersections in this view
  int n_points; ///< Number of quadrature points per intersection
  using IndexView =
      Kokkos::View<int *, Layout, ExecutionSpace>; ///< View type for integer
                                                   ///< indices
  using QPView =
      Kokkos::View<int **, Layout, ExecutionSpace>; ///< View type for
                                                    ///< quadrature point
                                                    ///< arrays
  using EdgeTypeView = ///< View type for 2D edge classifications
      Kokkos::View<specfem::mesh_entity::dim2::type *, ExecutionSpace>;

  using memory_space = typename ExecutionSpace::memory_space;

  using HostMirror =
      std::conditional_t<std::is_same<typename ExecutionSpace::memory_space,
                                      Kokkos::HostSpace>::value,
                         EdgeView,
                         EdgeView<Kokkos::DefaultHostExecutionSpace, Layout> >;

  /**
   * @brief Default constructor creating empty edge view.
   */
  EdgeView() : N(0), n_points(0) {}

  /**
   * @brief Construct edge view with allocated storage.
   *
   * @param label Base label for Kokkos view names
   * @param N Number of intersections to allocate
   * @param n_points Number of quadrature points per intersection
   */
  EdgeView(const std::string &label, const int N, const int n_points)
      : N(N), n_points(n_points), element_index(label + "_element_index", N),
        edge_index(label + "_edge_index", N),
        edge_types(label + "_edge_types", N), iz(label + "_iz", N, n_points),
        ix(label + "_ix", N, n_points) {}

  IndexView element_index; ///< Element indices for each intersection
  IndexView edge_index;    ///< Local edge indices within elements
  EdgeTypeView edge_types; ///< 2D edge type classifications
  QPView iz; ///< Z-direction quadrature indices for all intersections
  QPView ix; ///< X-direction quadrature indices for all edges

  /**
   * @brief Device-side constructor from existing views.
   *
   * @param N Number of edges
   * @param n_points Number of quadrature points per edge
   * @param element_index Element indices view
   * @param edge_index Edge indices view
   * @param edge_types Edge types view
   * @param iz Z-direction quadrature indices
   * @param ix X-direction quadrature indices
   */
  KOKKOS_INLINE_FUNCTION
  EdgeView(const int N, const int n_points, const IndexView &element_index,
           const IndexView &edge_index, const EdgeTypeView &edge_types,
           const QPView &iz, const QPView &ix)
      : N(N), n_points(n_points), element_index(element_index),
        edge_index(edge_index), edge_types(edge_types), iz(iz), ix(ix) {}

  /**
   * @brief Access individual edge by index.
   *
   * @param edge_id Index of the edge to access
   * @return Edge structure for the specified edge
   */
  KOKKOS_INLINE_FUNCTION
  Edge<ExecutionSpace> operator()(const int edge_id) const {
    return { n_points,
             element_index(edge_id),
             edge_index(edge_id),
             edge_types(edge_id),
             Kokkos::subview(iz, edge_id, Kokkos::ALL()),
             Kokkos::subview(ix, edge_id, Kokkos::ALL()) };
  }

  /**
   * @brief Access subrange of edges.
   *
   * @param edge_range Pair specifying start and end indices
   * @return EdgeView containing the specified range of edges
   */
  KOKKOS_INLINE_FUNCTION
  EdgeView<ExecutionSpace>
  operator()(const Kokkos::pair<int, int> &edge_range) const {
    return { edge_range.second - edge_range.first,
             n_points,
             Kokkos::subview(element_index, edge_range),
             Kokkos::subview(edge_index, edge_range),
             Kokkos::subview(edge_types, edge_range),
             Kokkos::subview(iz, edge_range, Kokkos::ALL()),
             Kokkos::subview(ix, edge_range, Kokkos::ALL()) };
  }

  /**
   * @brief Get total number of quadrature points across all edges in this view.
   *
   * @return Total number of quadrature points
   */
  KOKKOS_FORCEINLINE_FUNCTION
  int get_total_points() const { return N * n_points; }

  std::string label() const {
    const auto &raw = element_index.label();
    return raw.substr(0, raw.rfind("_element_index"));
  }
};

} // namespace specfem::assembly

namespace specfem::tag_dispatch {

template <typename DstSpace, typename SrcSpace, typename Layout>
auto create_mirror(DstSpace,
                   const specfem::assembly::EdgeView<SrcSpace, Layout> &src) {
  return specfem::assembly::EdgeView<DstSpace, Layout>(src.label(), src.N,
                                                       src.n_points);
}

template <typename DstSpace, typename SrcSpace, typename Layout>
void deep_copy(specfem::assembly::EdgeView<DstSpace, Layout> &dst,
               const specfem::assembly::EdgeView<SrcSpace, Layout> &src) {
  Kokkos::deep_copy(dst.element_index, src.element_index);
  Kokkos::deep_copy(dst.edge_index, src.edge_index);
  Kokkos::deep_copy(dst.edge_types, src.edge_types);
  Kokkos::deep_copy(dst.iz, src.iz);
  Kokkos::deep_copy(dst.ix, src.ix);
}

} // namespace specfem::tag_dispatch

namespace specfem::assembly {

/**
 * @brief 2D spectral element edge classification and coupling management
 *
 * This template specialization provides storage and management for edge
 * information in 2D spectral element meshes. It handles edge connectivity,
 * interface types, and boundary conditions for coupling between different
 * media types in 2D wave propagation problems.
 *
 * @code
 * // Construct 2D edge types from mesh data
 * specfem::assembly::element_intersections<specfem::element::dimension_tag::dim2>
 * edges( ngllx, ngllz, mesh, element_types);
 *
 * // Get elastic-acoustic coupling edges on device
 * auto [self_edges, coupled_edges] = edges.get_edges_on_device(
 *     specfem::element_connections::type::weakly_conforming,
 *     specfem::element_coupling::interface_tag::elastic_acoustic,
 *     specfem::element::boundary_tag::none);
 * @endcode
 */
template <>
struct element_intersections<specfem::element::dimension_tag::dim2> {

public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim2; ///< Dimension tag

public:
  /**
   * @brief Type alias for storing 2D edge collections in device memory.
   *
   */
  using EdgeViewType = EdgeView<Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Get edge pairs for coupling computations in host memory.
   *
   * @param connection Connection type (weakly_conforming, nonconforming)
   * @param edge Interface type (elastic_acoustic, acoustic_elastic)
   * @param boundary Boundary condition type
   * @return Tuple of (self_edges, coupled_edges) for host processing
   */
  std::tuple<typename EdgeViewType::HostMirror,
             typename EdgeViewType::HostMirror>
  get_intersections_on_host(
      const specfem::element_connections::type connection,
      const specfem::element_coupling::interface_tag edge,
      const specfem::element::boundary_tag boundary,
      const specfem::element_coupling::flux_scheme_tag flux_scheme) const;

  /**
   * @brief Get edge pairs for coupling computations in device memory.
   *
   * @param connection Connection type (weakly_conforming, nonconforming)
   * @param edge Interface type (elastic_acoustic, acoustic_elastic)
   * @param boundary Boundary condition type
   * @return Tuple of (self_edges, coupled_edges) for device processing
   */
  std::tuple<EdgeViewType, EdgeViewType> get_intersections_on_device(
      const specfem::element_connections::type connection,
      const specfem::element_coupling::interface_tag edge,
      const specfem::element::boundary_tag boundary,
      const specfem::element_coupling::flux_scheme_tag flux_scheme) const;

  /**
   * @brief Construct 2D edge types from mesh and element information.
   *
   * @param ngllx Number of quadrature points in x-direction
   * @param ngllz Number of quadrature points in z-direction
   * @param mesh 2D assembly mesh with connectivity information
   * @param element_types Element classification for coupling detection
   * @param flux_scheme_config Flux-scheme classification for proper
   * flux-tagging
   */
  element_intersections(
      const int ngllx, const int ngllz,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::element_coupling::flux_scheme_configuration
          &flux_scheme_config =
              specfem::element_coupling::flux_scheme_configuration());

  /**
   * @brief Default constructor.
   */
  element_intersections() = default;

private:
  // clang-format off
  using IntersectionCombinations = decltype(
           DIMENSION_SET(dim2) *
           CONNECTION_SET(weakly_conforming, nonconforming) *
           INTERFACE_SET(elastic_acoustic, acoustic_elastic) *
           BOUNDARY_SET(none, acoustic_free_surface, stacey,
                        composite_stacey_dirichlet) *
           FLUX_SCHEME_SET(natural, symmetric_interior_penalty));
  // clang-format on

  specfem::tag_dispatch::Storage<EdgeViewType, IntersectionCombinations>
      self_edges;
  specfem::tag_dispatch::Storage<EdgeViewType::HostMirror,
                                 IntersectionCombinations>
      h_self_edges;
  specfem::tag_dispatch::Storage<EdgeViewType, IntersectionCombinations>
      coupled_edges;
  specfem::tag_dispatch::Storage<EdgeViewType::HostMirror,
                                 IntersectionCombinations>
      h_coupled_edges;
};

specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim2>::EdgeViewType::HostMirror
edge_view_from_collected_edges(
    const std::string &label,
    const std::vector<
        specfem::mesh_entity::edge<specfem::element::dimension_tag::dim2> >
        &collected_edges,
    const specfem::mesh_entity::element<specfem::element::dimension_tag::dim2>
        &element);

template <typename Space, typename ExecSpace, typename Layout>
auto create_mirror(Space, const EdgeView<ExecSpace, Layout> &src) {
  return EdgeView<Space, Layout>(src.label(), src.N, src.n_points);
}

template <typename DestExecSpace, typename SrcExecSpace, typename Layout>
void deep_copy(EdgeView<DestExecSpace, Layout> &dst,
               const EdgeView<SrcExecSpace, Layout> &src) {
  Kokkos::deep_copy(dst.element_index, src.element_index);
  Kokkos::deep_copy(dst.edge_index, src.edge_index);
  Kokkos::deep_copy(dst.edge_types, src.edge_types);
  Kokkos::deep_copy(dst.iz, src.iz);
  Kokkos::deep_copy(dst.ix, src.ix);
}

} // namespace specfem::assembly
