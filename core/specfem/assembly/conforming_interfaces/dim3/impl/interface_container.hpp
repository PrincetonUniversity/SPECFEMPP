
#pragma once

#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/enums.hpp"

namespace specfem::assembly::conforming_interfaces_impl {

/**
 * @brief Container for 3D coupled interface data storage and access
 *
 * Manages interface data between different physical media (elastic-acoustic)
 * with specific boundary conditions. Stores face factors and normal vectors
 * for interface computations in 3D spectral element simulations.
 *
 * @tparam InterfaceTag Type of interface (ELASTIC_ACOUSTIC or ACOUSTIC_ELASTIC)
 * @tparam BoundaryTag Boundary condition type (NONE, STACEY, etc.)
 */
template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct interface_container<
    specfem::element::dimension_tag::dim3, InterfaceTag, BoundaryTag,
    specfem::element_connections::type::weakly_conforming>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::face,
          specfem::data_access::DataClassType::conforming_interface,
          specfem::element::dimension_tag::dim3> {
public:
  /** @brief Dimension tag for 3D specialization */
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  /** @brief Interface type (elastic-acoustic or acoustic-elastic) */
  constexpr static auto interface_tag = InterfaceTag;
  /** @brief Boundary condition type */
  constexpr static auto boundary_tag = BoundaryTag;
  /** @brief Medium type on the self side of the interface */
  constexpr static auto self_medium =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::self_medium();
  /** @brief Medium type on the coupled side of the interface */
  constexpr static auto coupled_medium =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::coupled_medium();

private:
  /** @brief Base container type alias */
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::face,
      specfem::data_access::DataClassType::conforming_interface,
      specfem::element::dimension_tag::dim3>;
  /** @brief View type for face scaling factors: (nfaces, npoint_i, npoint_j) */
  using FaceFactorView = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  /** @brief View type for face normal vectors: (nfaces, npoint_i, npoint_j, 3)
   */
  using FaceNormalView = typename base_type::tensor_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  /** @brief Device view for face scaling factors */
  FaceFactorView face_factor;
  /** @brief Device view for face normal vectors */
  FaceNormalView face_normal;

  /** @brief Host mirror for face scaling factors */
  FaceFactorView::HostMirror h_face_factor;
  /** @brief Host mirror for face normal vectors */
  FaceNormalView::HostMirror h_face_normal;

public:
  /**
   * @brief Constructs interface container with mesh and geometry data
   *
   * @param ngllz Number of GLL points in z-direction
   * @param nglly Number of GLL points in y-direction
   * @param ngllx Number of GLL points in x-direction
   * @param element_intersections Element intersection information from mesh
   * @param jacobian_matrix Jacobian transformation data
   * @param mesh Mesh connectivity and geometry
   */
  interface_container(
      const int ngllz, const int nglly, const int ngllx,
      const specfem::assembly::element_intersections<
          specfem::element::dimension_tag::dim3> &element_intersections,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::mesh<dimension_tag> &mesh);

  /** @brief Default constructor */
  interface_container() = default;

  /**
   * @brief Loads interface data at specified index into point
   *
   * Template function that loads face factor and normal vector data
   * from either device or host memory into the provided point object.
   *
   * @tparam on_device If true, loads from device memory; if false, from host
   * @tparam IndexType Type of index (must have iface and ipoint members)
   * @tparam PointType Type of point (must have face_factor and face_normal)
   * @param index Face and point indices for data location
   * @param point Output point object to store loaded data
   */
  template <bool on_device, typename IndexType, typename PointType>
  KOKKOS_FORCEINLINE_FUNCTION void
  impl_load(const std::integral_constant<
                specfem::datatype::AccessorType,
                specfem::datatype::AccessorType::point> /* AccessorType */,
            const IndexType &index, PointType &point) const {

    static_assert(specfem::data_access::is_point<PointType>::value,
                  "impl_load only supports point accessors");

    static_assert(specfem::data_access::is_point<IndexType>::value,
                  "impl_load requires point type for IndexType");

    static_assert(specfem::data_access::is_face_index<IndexType>::value,
                  "impl_load requires face index type for IndexType");

    static_assert(
        specfem::data_access::is_conforming_interface<PointType>::value,
        "impl_load requires conforming interface point type for PointType");

    if constexpr (on_device) {
      point.face_factor =
          face_factor(index.iface, index.ipoint_i, index.ipoint_j);
      point.face_normal(0) =
          face_normal(index.iface, index.ipoint_i, index.ipoint_j, 0);
      point.face_normal(1) =
          face_normal(index.iface, index.ipoint_i, index.ipoint_j, 1);
      point.face_normal(2) =
          face_normal(index.iface, index.ipoint_i, index.ipoint_j, 2);
    } else {
      point.face_factor =
          h_face_factor(index.iface, index.ipoint_i, index.ipoint_j);
      point.face_normal(0) =
          h_face_normal(index.iface, index.ipoint_i, index.ipoint_j, 0);
      point.face_normal(1) =
          h_face_normal(index.iface, index.ipoint_i, index.ipoint_j, 1);
      point.face_normal(2) =
          h_face_normal(index.iface, index.ipoint_i, index.ipoint_j, 2);
    }
    return;
  }
};
} // namespace specfem::assembly::conforming_interfaces_impl
