#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/element.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/element_coupling/flux_scheme_configuration.hpp"
#include "specfem/mesh.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Individual 3D face representation with quadrature point access
 *
 * This structure represents a single face in a 3D spectral element mesh,
 * providing access to quadrature points on the face for coupling
 * computations, boundary condition enforcement, and flux calculations.
 *
 * @tparam ExecutionSpace Kokkos execution space (host or device)
 */
template <typename ExecutionSpace> struct Face {
  int n_points; ///< Number of quadrature points per face dimension
  using IndexView = Kokkos::View<int **, Kokkos::LayoutStride,
                                 ExecutionSpace>; ///< View type for quadrature
                                                  ///< indices
  int element_index; ///< Index of the spectral element containing this face
  int face_index;    ///< Local face index within the element
  specfem::mesh_entity::dim3::type face_type; ///< 3D face type (boundary
                                              ///< classification)
  IndexView iz; ///< Quadrature point indices in z-direction
  IndexView iy; ///< Quadrature point indices in y-direction
  IndexView ix; ///< Quadrature point indices in x-direction

  /**
   * @brief Construct face with quadrature point information.
   *
   * @param n_points Number of quadrature points per face dimension
   * @param element_index Element containing this face
   * @param face_index Local face index within element
   * @param face_type 3D face classification type
   * @param iz Z-direction quadrature indices
   * @param iy Y-direction quadrature indices
   * @param ix X-direction quadrature indices
   */
  KOKKOS_INLINE_FUNCTION
  Face(const int n_points, const int element_index, const int face_index,
       const specfem::mesh_entity::dim3::type face_type, const IndexView iz,
       const IndexView iy, const IndexView ix)
      : n_points(n_points), element_index(element_index),
        face_index(face_index), face_type(face_type), iz(iz), iy(iy), ix(ix) {}

  /**
   * @brief Access quadrature point on the face.
   *
   * @param ipoint_i First face coordinate index (0 to n_points-1)
   * @param ipoint_j Second face coordinate index (0 to n_points-1)
   * @return 3D face index structure for the specified quadrature point
   */
  KOKKOS_INLINE_FUNCTION
  specfem::point::face_index<specfem::element::dimension_tag::dim3>
  operator()(const int ipoint_i, const int ipoint_j) const {
    return { element_index,
             face_index,
             ipoint_i,
             ipoint_j,
             iz(ipoint_i, ipoint_j),
             iy(ipoint_i, ipoint_j),
             ix(ipoint_i, ipoint_j),
             face_type };
  }
};

/**
 * @brief Collection of 3D faces with parallel access capabilities
 *
 * This structure manages collections of faces for efficient parallel
 * processing of face-based operations such as coupling computations,
 * boundary condition enforcement, and flux calculations in 3D.
 *
 * @tparam ExecutionSpace Kokkos execution space (host or device)
 * @tparam Layout Memory layout for Kokkos views
 */
template <typename ExecutionSpace,
          typename Layout = typename ExecutionSpace::array_layout>
struct FaceView {
  int N;        ///< Number of faces in this view
  int n_points; ///< Number of quadrature points per face dimension
  using IndexView =
      Kokkos::View<int *, Layout, ExecutionSpace>; ///< View type for integer
                                                   ///< indices
  using QPView =
      Kokkos::View<int ***, Layout, ExecutionSpace>; ///< View type for
                                                     ///< quadrature point
                                                     ///< arrays (3D: face x i x
                                                     ///< j)
  using FaceTypeView = ///< View type for 3D face classifications
      Kokkos::View<specfem::mesh_entity::dim3::type *, ExecutionSpace>;

  using memory_space = typename ExecutionSpace::memory_space;

  using HostMirror =
      std::conditional_t<std::is_same<typename ExecutionSpace::memory_space,
                                      Kokkos::HostSpace>::value,
                         FaceView,
                         FaceView<Kokkos::DefaultHostExecutionSpace, Layout> >;

  /**
   * @brief Default constructor creating empty face view.
   */
  FaceView() : N(0), n_points(0) {}

  /**
   * @brief Construct face view with allocated storage.
   *
   * @param label Base label for Kokkos view names
   * @param N Number of faces to allocate
   * @param n_points Number of quadrature points per face dimension
   */
  FaceView(const std::string &label, const int N, const int n_points)
      : N(N), n_points(n_points), element_index(label + "_element_index", N),
        face_index(label + "_face_index", N),
        face_types(label + "_face_types", N),
        iz(label + "_iz", N, n_points, n_points),
        iy(label + "_iy", N, n_points, n_points),
        ix(label + "_ix", N, n_points, n_points) {}

  IndexView element_index; ///< Element indices for each face
  IndexView face_index;    ///< Local face indices within elements
  FaceTypeView face_types; ///< 3D face type classifications
  QPView iz; ///< Z-direction quadrature indices for all intesections
  QPView iy; ///< Y-direction quadrature indices for all intesections
  QPView ix; ///< X-direction quadrature indices for all intesections

  /**
   * @brief Device-side constructor from existing views.
   *
   * @param N Number of intesections to allocate
   * @param n_points Number of quadrature points per face dimension
   * @param element_index Element indices view
   * @param face_index Face indices view
   * @param face_types Face types view
   * @param iz Z-direction quadrature indices
   * @param iy Y-direction quadrature indices
   * @param ix X-direction quadrature indices
   */
  KOKKOS_INLINE_FUNCTION
  FaceView(const int N, const int n_points, const IndexView &element_index,
           const IndexView &face_index, const FaceTypeView &face_types,
           const QPView &iz, const QPView &iy, const QPView &ix)
      : N(N), n_points(n_points), element_index(element_index),
        face_index(face_index), face_types(face_types), iz(iz), iy(iy), ix(ix) {
  }

  /**
   * @brief Access individual face by index.
   *
   * @param face_id Index of the face to access
   * @return Face structure for the specified face
   */
  KOKKOS_INLINE_FUNCTION
  Face<ExecutionSpace> operator()(const int face_id) const {
    return { n_points,
             element_index(face_id),
             face_index(face_id),
             face_types(face_id),
             Kokkos::subview(iz, face_id, Kokkos::ALL(), Kokkos::ALL()),
             Kokkos::subview(iy, face_id, Kokkos::ALL(), Kokkos::ALL()),
             Kokkos::subview(ix, face_id, Kokkos::ALL(), Kokkos::ALL()) };
  }

  /**
   * @brief Access subrange of faces.
   *
   * @param face_range Pair specifying start and end indices
   * @return FaceView containing the specified range of faces
   */
  KOKKOS_INLINE_FUNCTION
  FaceView<ExecutionSpace>
  operator()(const Kokkos::pair<int, int> &face_range) const {
    return { face_range.second - face_range.first,
             n_points,
             Kokkos::subview(element_index, face_range),
             Kokkos::subview(face_index, face_range),
             Kokkos::subview(face_types, face_range),
             Kokkos::subview(iz, face_range, Kokkos::ALL(), Kokkos::ALL()),
             Kokkos::subview(iy, face_range, Kokkos::ALL(), Kokkos::ALL()),
             Kokkos::subview(ix, face_range, Kokkos::ALL(), Kokkos::ALL()) };
  }

  /**
   * @brief Get total number of quadrature points across all faces in this view.
   *
   * @return Total number of quadrature points
   */
  KOKKOS_FORCEINLINE_FUNCTION
  int get_total_points() const { return N * n_points * n_points; }

  std::string label() const {
    const auto &raw = element_index.label();
    return raw.substr(0, raw.rfind("_element_index"));
  }
};

} // namespace specfem::assembly

namespace specfem::tag_dispatch {

template <typename DstSpace, typename SrcSpace, typename Layout>
auto create_mirror(DstSpace,
                   const specfem::assembly::FaceView<SrcSpace, Layout> &src) {
  return specfem::assembly::FaceView<DstSpace, Layout>(src.label(), src.N,
                                                       src.n_points);
}

template <typename DstSpace, typename SrcSpace, typename Layout>
void deep_copy(specfem::assembly::FaceView<DstSpace, Layout> &dst,
               const specfem::assembly::FaceView<SrcSpace, Layout> &src) {
  Kokkos::deep_copy(dst.element_index, src.element_index);
  Kokkos::deep_copy(dst.face_index, src.face_index);
  Kokkos::deep_copy(dst.face_types, src.face_types);
  Kokkos::deep_copy(dst.iz, src.iz);
  Kokkos::deep_copy(dst.iy, src.iy);
  Kokkos::deep_copy(dst.ix, src.ix);
}

} // namespace specfem::tag_dispatch

namespace specfem::assembly {

/**
 * @brief 3D spectral element face classification and coupling management
 *
 * This template specialization provides storage and management for face
 * information in 3D spectral element meshes. It handles face connectivity,
 * interface types, and boundary conditions for coupling between different
 * media types in 3D wave propagation problems.
 *
 * @code
 * // Construct 3D face types from mesh data
 * specfem::assembly::element_intersections<specfem::element::dimension_tag::dim3>
 * faces( ngllx, nglly, ngllz, mesh, element_types);
 *
 * // Get elastic-acoustic coupling faces on device
 * auto [self_faces, coupled_faces] = faces.get_faces_on_device(
 *     specfem::element_connections::type::weakly_conforming,
 *     specfem::element_coupling::interface_tag::elastic_acoustic,
 *     specfem::element::boundary_tag::none);
 * @endcode
 */
template <>
struct element_intersections<specfem::element::dimension_tag::dim3> {

public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag

public:
  /**
   * @brief Type alias for storing 3D face collections in device memory.
   */
  using FaceViewType = FaceView<Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Get face pairs for coupling computations in host memory.
   *
   * @param connection Connection type (weakly_conforming, nonconforming)
   * @param face Interface type (elastic_acoustic, acoustic_elastic)
   * @param boundary Boundary condition type
   * @return Tuple of (self_faces, coupled_faces) for host processing
   */
  std::tuple<typename FaceViewType::HostMirror,
             typename FaceViewType::HostMirror>
  get_intersections_on_host(
      const specfem::element_connections::type connection,
      const specfem::element_coupling::interface_tag face,
      const specfem::element::boundary_tag boundary,
      const specfem::element_coupling::flux_scheme_tag flux_scheme) const;

  /**
   * @brief Get face pairs for coupling computations in device memory.
   *
   * @param connection Connection type (weakly_conforming, nonconforming)
   * @param face Interface type (elastic_acoustic, acoustic_elastic)
   * @param boundary Boundary condition type
   * @return Tuple of (self_faces, coupled_faces) for device processing
   */
  std::tuple<FaceViewType, FaceViewType> get_intersections_on_device(
      const specfem::element_connections::type connection,
      const specfem::element_coupling::interface_tag face,
      const specfem::element::boundary_tag boundary,
      const specfem::element_coupling::flux_scheme_tag flux_scheme) const;

  /**
   * @brief Construct 3D face types from mesh and element information.
   *
   * @param ngllx Number of quadrature points in x-direction
   * @param nglly Number of quadrature points in y-direction
   * @param ngllz Number of quadrature points in z-direction
   * @param mesh 3D assembly mesh with connectivity information
   * @param element_types Element classification for coupling detection
   */
  element_intersections(
      const int ngllx, const int nglly, const int ngllz,
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
           DIMENSION_SET(dim3) *
           CONNECTION_SET(weakly_conforming, nonconforming) *
           INTERFACE_SET(elastic_acoustic, acoustic_elastic) *
           BOUNDARY_SET(none, acoustic_free_surface, stacey,
                        composite_stacey_dirichlet) *
           FLUX_SCHEME_SET(natural));
  // clang-format on

  specfem::tag_dispatch::Storage<FaceViewType, IntersectionCombinations>
      self_faces;
  specfem::tag_dispatch::Storage<FaceViewType::HostMirror,
                                 IntersectionCombinations>
      h_self_faces;
  specfem::tag_dispatch::Storage<FaceViewType, IntersectionCombinations>
      coupled_faces;
  specfem::tag_dispatch::Storage<FaceViewType::HostMirror,
                                 IntersectionCombinations>
      h_coupled_faces;
};

/**
 * @brief Build a host-side FaceView from a collection of 3D face entities.
 *
 * Constructs and fills a host-memory FaceView by mapping each collected
 * face's quadrature point indices using the provided element coordinate
 * mapping.
 *
 * @param label Base label for Kokkos view names
 * @param collected_faces Vector of 3D face entities to convert
 * @param element 3D element providing coordinate mapping
 * @return Populated host-mirror FaceView
 */
specfem::assembly::element_intersections<
    specfem::element::dimension_tag::dim3>::FaceViewType::HostMirror
face_view_from_collected_faces(
    const std::string &label,
    const std::vector<
        specfem::mesh_entity::face<specfem::element::dimension_tag::dim3> >
        &collected_faces,
    const specfem::mesh_entity::element<specfem::element::dimension_tag::dim3>
        &element);

template <typename Space, typename ExecSpace, typename Layout>
auto create_mirror(Space, const FaceView<ExecSpace, Layout> &src) {
  return FaceView<Space, Layout>(src.label(), src.N, src.n_points);
}

template <typename DestExecSpace, typename SrcExecSpace, typename Layout>
void deep_copy(FaceView<DestExecSpace, Layout> &dst,
               const FaceView<SrcExecSpace, Layout> &src) {
  Kokkos::deep_copy(dst.element_index, src.element_index);
  Kokkos::deep_copy(dst.face_index, src.face_index);
  Kokkos::deep_copy(dst.face_types, src.face_types);
  Kokkos::deep_copy(dst.iz, src.iz);
  Kokkos::deep_copy(dst.iy, src.iy);
  Kokkos::deep_copy(dst.ix, src.ix);
}

} // namespace specfem::assembly
