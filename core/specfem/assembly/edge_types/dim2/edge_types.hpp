#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>

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
 * specfem::assembly::edge_types<specfem::dimension::type::dim2> edges(
 *     ngllx, ngllz, mesh, element_types);
 *
 * // Get elastic-acoustic coupling edges on device
 * auto [self_edges, coupled_edges] = edges.get_edges_on_device(
 *     specfem::connections::type::weakly_conforming,
 *     specfem::interface::interface_tag::elastic_acoustic,
 *     specfem::element::boundary_tag::none);
 * @endcode
 */
template <> struct edge_types<specfem::dimension::type::dim2> {

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag

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
    specfem::point::edge_index<specfem::dimension::type::dim2>
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
    int n_edges;  ///< Number of edges in this view
    int n_points; ///< Number of quadrature points per edge
    using IndexView =
        Kokkos::View<int *, Layout, ExecutionSpace>; ///< View type for integer
                                                     ///< indices
    using QPView =
        Kokkos::View<int **, Layout, ExecutionSpace>; ///< View type for
                                                      ///< quadrature point
                                                      ///< arrays
    using EdgeTypeView = ///< View type for 2D edge classifications
        Kokkos::View<specfem::mesh_entity::dim2::type *, ExecutionSpace>;

    using HostMirror = std::conditional_t<
        std::is_same<typename ExecutionSpace::memory_space,
                     Kokkos::HostSpace>::value,
        EdgeView, EdgeView<Kokkos::DefaultHostExecutionSpace, Layout> >;

    /**
     * @brief Default constructor creating empty edge view.
     */
    EdgeView() : n_edges(0), n_points(0) {}

    /**
     * @brief Construct edge view with allocated storage.
     *
     * @param label Base label for Kokkos view names
     * @param n_edges Number of edges to allocate
     * @param n_points Number of quadrature points per edge
     */
    EdgeView(const std::string &label, const int n_edges, const int n_points)
        : n_edges(n_edges), n_points(n_points),
          element_index(label + "_element_index", n_edges),
          edge_index(label + "_edge_index", n_edges),
          edge_types(label + "_edge_types", n_edges),
          iz(label + "_iz", n_edges, n_points),
          ix(label + "_ix", n_edges, n_points) {}

    IndexView element_index; ///< Element indices for each edge
    IndexView edge_index;    ///< Local edge indices within elements
    EdgeTypeView edge_types; ///< 2D edge type classifications
    QPView iz;               ///< Z-direction quadrature indices for all edges
    QPView ix;               ///< X-direction quadrature indices for all edges

    /**
     * @brief Device-side constructor from existing views.
     *
     * @param n_edges Number of edges
     * @param n_points Number of quadrature points per edge
     * @param element_index Element indices view
     * @param edge_index Edge indices view
     * @param edge_types Edge types view
     * @param iz Z-direction quadrature indices
     * @param ix X-direction quadrature indices
     */
    KOKKOS_INLINE_FUNCTION
    EdgeView(const int n_edges, const int n_points,
             const IndexView &element_index, const IndexView &edge_index,
             const EdgeTypeView &edge_types, const QPView &iz, const QPView &ix)
        : n_edges(n_edges), n_points(n_points), element_index(element_index),
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
  };

public:
  /**
   * @brief Type alias for storing 2D edge collections in device memory.
   *
   */
  using EdgeViewType = EdgeView<Kokkos::DefaultExecutionSpace>;

private:
  static EdgeViewType::HostMirror create_mirror_view(const EdgeViewType &view) {
    const auto label = view.element_index.label();
    // remove element_index suffix
    const auto base_label = label.substr(0, label.size() - 14);
    return EdgeViewType::HostMirror(base_label + "_host_mirror", view.n_edges,
                                    view.n_points);
  }

  template <typename SrcView, typename DestView>
  static void deep_copy(const DestView &dest, const SrcView &src) {
    Kokkos::deep_copy(dest.element_index, src.element_index);
    Kokkos::deep_copy(dest.edge_index, src.edge_index);
    Kokkos::deep_copy(dest.edge_types, src.edge_types);
    Kokkos::deep_copy(dest.iz, src.iz);
    Kokkos::deep_copy(dest.ix, src.ix);
  }

public:
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
  get_edges_on_host(const specfem::connections::type connection,
                    const specfem::interface::interface_tag edge,
                    const specfem::element::boundary_tag boundary) const;

  /**
   * @brief Get edge pairs for coupling computations in device memory.
   *
   * @param connection Connection type (weakly_conforming, nonconforming)
   * @param edge Interface type (elastic_acoustic, acoustic_elastic)
   * @param boundary Boundary condition type
   * @return Tuple of (self_edges, coupled_edges) for device processing
   */
  std::tuple<EdgeViewType, EdgeViewType>
  get_edges_on_device(const specfem::connections::type connection,
                      const specfem::interface::interface_tag edge,
                      const specfem::element::boundary_tag boundary) const;

  /**
   * @brief Construct 2D edge types from mesh and element information.
   *
   * @param ngllx Number of quadrature points in x-direction
   * @param ngllz Number of quadrature points in z-direction
   * @param mesh 2D assembly mesh with connectivity information
   * @param element_types Element classification for coupling detection
   */
  edge_types(
      const int ngllx, const int ngllz,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types);

  /**
   * @brief Default constructor.
   */
  edge_types() = default;

private:
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
                       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
                       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      DECLARE((EdgeViewType, self_edges),
                              (EdgeViewType::HostMirror, h_self_edges),
                              (EdgeViewType, coupled_edges),
                              (EdgeViewType::HostMirror, h_coupled_edges)))
};

} // namespace specfem::assembly
