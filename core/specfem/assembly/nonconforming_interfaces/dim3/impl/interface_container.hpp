#pragma once

#include "specfem/assembly/nonconforming_interfaces/fwd.hpp"

#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/element/dimension.hpp"
#include "specfem/element_coupling/flux_scheme_configuration.hpp"
#include "specfem/element_coupling/tags.hpp"
#include "specfem/enums.hpp"
#include "specfem/execution.hpp"

namespace specfem::assembly::nonconforming_interfaces_impl {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_connections::type ConnectionTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
struct interface_container;

/**
 * @brief Container for 3D nonconforming interface data storage and access
 *
 * Manages interface data between different physical media (elastic-acoustic)
 * with specific boundary conditions. Stores edge factors and normal vectors
 * for interface computations in 3D spectral element simulations.
 *
 * TODO: consider same physical media
 *
 * @tparam InterfaceTag Type of interface (ELASTIC_ACOUSTIC or ACOUSTIC_ELASTIC)
 * @tparam BoundaryTag Boundary condition type (NONE, STACEY, etc.)
 */
template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
struct interface_container<
    specfem::element::dimension_tag::dim3, InterfaceTag, BoundaryTag,
    specfem::element_connections::type::nonconforming, FluxSchemeTag>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::face,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::element::dimension_tag::dim3> {
public:
  /** @brief Dimension tag for 2D specialization */
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  /** @brief Interface type (elastic-acoustic or acoustic-elastic) */
  constexpr static auto interface_tag = InterfaceTag;
  /** @brief Boundary condition type */
  constexpr static auto boundary_tag = BoundaryTag;
  /** @brief Flux scheme type */
  constexpr static auto flux_scheme_tag = FluxSchemeTag;
  /** @brief Medium type on the self side of the interface */
  constexpr static auto self_medium =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::self_medium();
  /** @brief Medium type on the coupled side of the interface */
  constexpr static auto coupled_medium =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::coupled_medium();

public:
  /** @brief Base container type alias */
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::face,
      specfem::data_access::DataClassType::nonconforming_interface,
      specfem::element::dimension_tag::dim3>;
  /** @brief View type for edge scaling factors */
  using FaceFactorView = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  /** @brief View type for edge normal vectors */
  using FaceNormalView = typename base_type::tensor_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  /** @brief View type for transfer function */
  using CoupledCoordinatesView = typename base_type::tensor_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  /** @brief Device view for edge scaling factors */
  FaceFactorView face_factor;
  /** @brief Device view for edge normal vectors */
  FaceNormalView face_normal;
  /** @brief Device view for self nodes in coupled coordinates */
  CoupledCoordinatesView coupled_coordinates;

  /** @brief Host mirror for edge scaling factors */
  FaceFactorView::host_mirror_type h_face_factor;
  /** @brief Host mirror for edge normal vectors */
  FaceNormalView::host_mirror_type h_face_normal;
  /** @brief Device view for self nodes in coupled coordinates */
  CoupledCoordinatesView::host_mirror_type h_coupled_coordinates;

public:
  /**
   * @brief Constructs interface container with mesh and geometry data
   *
   * @param ngllz Number of GLL points in z-direction
   * @param ngllx Number of GLL points in x-direction
   * @param element_intersections Element intersection information from mesh
   * @param mesh Mesh connectivity and geometry
   */
  interface_container(
      const int &ngllz, const int &nglly, const int &ngllx,
      const specfem::assembly::element_intersections<
          specfem::element::dimension_tag::dim3> &element_intersections,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::element_coupling::flux_scheme_configuration
          &flux_scheme_config = {});

  /** @brief Default constructor */
  interface_container() = default;

  interface_container(const int &ngllz, const int &nglly, const int &ngllx,
                      const int &num_faces)
      : face_factor("specfem::assembly::nonconforming_interfaces::face_factor",
                    num_faces, std::max(std::max(ngllz, nglly), ngllx),
                    std::max(std::max(ngllz, nglly), ngllx)),
        h_face_factor(Kokkos::create_mirror_view(face_factor)),
        face_normal("specfem::assembly::nonconforming_interfaces::face_normal",
                    num_faces, std::max(std::max(ngllz, nglly), ngllx),
                    std::max(std::max(ngllz, nglly), ngllx),
                    specfem::element::dimension<dimension_tag>::dim),
        h_face_normal(Kokkos::create_mirror_view(face_normal)),
        coupled_coordinates(
            "specfem::assembly::nonconforming_interfaces::coupled_coordinates",
            num_faces, std::max(std::max(ngllz, nglly), ngllx),
            std::max(std::max(ngllz, nglly), ngllx),
            specfem::element::dimension<dimension_tag>::dim - 1),
        h_coupled_coordinates(Kokkos::create_mirror_view(coupled_coordinates)) {
        };

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
        specfem::data_access::is_nonconforming_interface<PointType>::value,
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
      point.coupled_coordinates(0) =
          coupled_coordinates(index.iface, index.ipoint_i, index.ipoint_j, 0);
      point.coupled_coordinates(1) =
          coupled_coordinates(index.iface, index.ipoint_i, index.ipoint_j, 1);
    } else {
      point.face_factor =
          h_face_factor(index.iface, index.ipoint_i, index.ipoint_j);
      point.face_normal(0) =
          h_face_normal(index.iface, index.ipoint_i, index.ipoint_j, 0);
      point.face_normal(1) =
          h_face_normal(index.iface, index.ipoint_i, index.ipoint_j, 1);
      point.face_normal(2) =
          h_face_normal(index.iface, index.ipoint_i, index.ipoint_j, 2);
      point.coupled_coordinates(0) =
          h_coupled_coordinates(index.iface, index.ipoint_i, index.ipoint_j, 0);
      point.coupled_coordinates(1) =
          h_coupled_coordinates(index.iface, index.ipoint_i, index.ipoint_j, 1);
    }
    return;
  }
};
} // namespace specfem::assembly::nonconforming_interfaces_impl
