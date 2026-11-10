#pragma once

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
 * with specific boundary conditions. Stores, per pair and self-face node, the
 * integration factor and normal vector and, when the self side is the host,
 * the guest-face Lagrange interpolants at that node.
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
  /** @brief Dimension tag for 3D specialization */
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
  /** @brief View type for the guest-face 1D interpolants at host nodes */
  using InterpolantsView =
      Kokkos::View<type_real *****, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace::memory_space>;

  /** @brief Device view for edge scaling factors */
  FaceFactorView face_factor{};
  /** @brief Device view for edge normal vectors */
  FaceNormalView face_normal{};
  /**
   * @brief Device view of the guest face's 1D Lagrange polynomials at each
   * host node: `(ipair, ipoint_i, ipoint_j, igll, axis)`.
   *
   * `interpolants(ipair, i, j, k, 0)` is \f$ \ell_k(\xi_c) \f$ and
   * `(..., 1)` is \f$ \ell_k(\eta_c) \f$, where \f$ (\xi_c, \eta_c) \f$
   * locates host node `(i, j)` of pair `ipair` on its guest face; axis 0
   * pairs with the guest's `ipoint_i`. Host nodes owned by another pair have
   * zero rows. Allocated only when `self_is_host`.
   */
  InterpolantsView interpolants{};

  /** @brief Host mirror for edge scaling factors */
  FaceFactorView::host_mirror_type h_face_factor{};
  /** @brief Host mirror for edge normal vectors */
  FaceNormalView::host_mirror_type h_face_normal{};
  /** @brief Host mirror of `interpolants` */
  InterpolantsView::host_mirror_type h_interpolants{};

  /**
   * @brief True when the self faces of this container are the quadrature
   * (host) side of their pairs; only then is `interpolants` allocated.
   */
  bool self_is_host = false;

public:
  /**
   * @brief Constructs interface container with mesh and geometry data
   *
   * @param ngllz Number of GLL points in z-direction
   * @param nglly Number of GLL points in y-direction
   * @param ngllx Number of GLL points in x-direction
   * @param element_intersections Element intersection information from mesh
   * @param jacobian_matrix Jacobian matrix used for the face normals
   * @param mesh Mesh connectivity and geometry
   * @param flux_scheme_config Selects which side of each pair is the host
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

  /**
   * @brief Loads interface data at specified index into point
   *
   * Loads the face factor, the normal vector and the guest-face
   * interpolants of one host node from either device or host memory.
   * The interpolants are read from `interpolants`, so this container must be
   * the host (`self_is_host`).
   *
   * @tparam on_device If true, loads from device memory; if false, from host
   * @tparam IndexType Type of index (must have iface and ipoint members)
   * @tparam PointType Type of point (must have face_factor, face_normal and
   * interpolants)
   * @param index Face and point indices for data location
   * @param point Output point object to store loaded data
   *
   * @pre `self_is_host` is true. A container that is not the host stores no
   * interpolants.
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
        "impl_load requires nonconforming interface point type for PointType");

    if constexpr (on_device) {
      point.face_factor =
          face_factor(index.iface, index.ipoint_i, index.ipoint_j);
      point.face_normal(0) =
          face_normal(index.iface, index.ipoint_i, index.ipoint_j, 0);
      point.face_normal(1) =
          face_normal(index.iface, index.ipoint_i, index.ipoint_j, 1);
      point.face_normal(2) =
          face_normal(index.iface, index.ipoint_i, index.ipoint_j, 2);
      for (int igll = 0; igll < PointType::ngll; igll++) {
        point.interpolants(igll, 0) =
            interpolants(index.iface, index.ipoint_i, index.ipoint_j, igll, 0);
        point.interpolants(igll, 1) =
            interpolants(index.iface, index.ipoint_i, index.ipoint_j, igll, 1);
      }
    } else {
      point.face_factor =
          h_face_factor(index.iface, index.ipoint_i, index.ipoint_j);
      point.face_normal(0) =
          h_face_normal(index.iface, index.ipoint_i, index.ipoint_j, 0);
      point.face_normal(1) =
          h_face_normal(index.iface, index.ipoint_i, index.ipoint_j, 1);
      point.face_normal(2) =
          h_face_normal(index.iface, index.ipoint_i, index.ipoint_j, 2);
      for (int igll = 0; igll < PointType::ngll; igll++) {
        point.interpolants(igll, 0) = h_interpolants(
            index.iface, index.ipoint_i, index.ipoint_j, igll, 0);
        point.interpolants(igll, 1) = h_interpolants(
            index.iface, index.ipoint_i, index.ipoint_j, igll, 1);
      }
    }
    return;
  }
};
} // namespace specfem::assembly::nonconforming_interfaces_impl
