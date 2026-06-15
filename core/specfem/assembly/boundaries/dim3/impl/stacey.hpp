#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Stacey absorbing boundary condition data container for 3D SEM
 *
 * Stores per-quadrature-point boundary tag, face normal vector, and face
 * integration weight data for elements with Stacey absorbing boundary
 * conditions. In 3D, boundaries are faces (not edges), so the geometric data
 * uses face_normal (3 components) and face_weight instead of edge_normal and
 * edge_weight.
 *
 */
template <> struct stacey<specfem::element::dimension_tag::dim3> {
private:
  constexpr static auto boundary_tag = specfem::element::boundary_tag::stacey;

public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  /**
   * @brief Kokkos view for boundary tags per quadrature point on device
   *
   * Dimensions: [nspec_stacey, ngllz, nglly, ngllx]
   */
  using BoundaryTagView =
      Kokkos::View<specfem::element::boundary_tag_container ****,
                   Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view for face normal vectors on device
   *
   * Dimensions: [nspec_stacey, ngllz, nglly, ngllx, 3]
   */
  using FaceNormalView = Kokkos::View<type_real *****, Kokkos::LayoutLeft,
                                      Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view for face integration weights on device
   *
   * Dimensions: [nspec_stacey, ngllz, nglly, ngllx]
   */
  using FaceWeightView = Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                      Kokkos::DefaultExecutionSpace>;

  BoundaryTagView quadrature_point_boundary_tag; ///< Device boundary tags
  BoundaryTagView::host_mirror_type
      h_quadrature_point_boundary_tag; ///< Host
                                       ///< boundary
                                       ///< tags

  FaceNormalView face_normal;                     ///< Device face normals
  FaceNormalView::host_mirror_type h_face_normal; ///< Host face normals

  FaceWeightView face_weight;                     ///< Device face weights
  FaceWeightView::host_mirror_type h_face_weight; ///< Host face weights

  stacey() = default;

  /**
   * @brief Construct from mesh boundary information
   *
   * @param nspec Total number of spectral elements
   * @param ngllz GLL points in z direction
   * @param nglly GLL points in y direction
   * @param ngllx GLL points in x direction
   * @param mesh Mesh with boundary face data
   * @param mesh_assembly Assembly mesh with coordinate and mapping info
   * @param jacobian_matrix Jacobian matrix for computing face normals/weights
   * @param boundary_index_mapping Mapping from compute element index to local
   *        stacey index (-1 if element has no stacey boundary)
   * @param boundary_tag Per-element boundary tag containers to update
   */
  stacey(
      const int nspec, const int ngllz, const int nglly, const int ngllx,
      const specfem::mesh::mesh<dimension_tag> &mesh,
      const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
      std::vector<specfem::element::boundary_tag_container> &boundary_tag);

  // ── Device load methods ──────────────────────────────────────────────────

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == false,
                int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const IndexType &index,
                 specfem::point::boundary<boundary_tag, dimension_tag, false>
                     &boundary) const {
    boundary.tag += quadrature_point_boundary_tag(index.ispec, index.iz,
                                                  index.iy, index.ix);
    boundary.face_normal(0) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 0);
    boundary.face_normal(1) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 1);
    boundary.face_normal(2) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 2);
    boundary.face_weight =
        face_weight(index.ispec, index.iz, index.iy, index.ix);
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == false,
                int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const IndexType &index,
                 specfem::point::boundary<
                     specfem::element::boundary_tag::composite_stacey_dirichlet,
                     dimension_tag, false> &boundary) const {
    boundary.tag += quadrature_point_boundary_tag(index.ispec, index.iz,
                                                  index.iy, index.ix);
    boundary.face_normal(0) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 0);
    boundary.face_normal(1) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 1);
    boundary.face_normal(2) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 2);
    boundary.face_weight =
        face_weight(index.ispec, index.iz, index.iy, index.ix);
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == true,
                int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const IndexType &index,
                 specfem::point::boundary<boundary_tag, dimension_tag, true>
                     &boundary) const {
    using simd = typename specfem::datatype::simd<type_real, true>;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const auto mask = index.template get_mask<simd>();

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }

    boundary.face_normal(0) = Kokkos::Experimental::simd_partial_load(
        &face_normal(index.ispec, index.iz, index.iy, index.ix, 0), mask,
        tag_type());
    boundary.face_normal(1) = Kokkos::Experimental::simd_partial_load(
        &face_normal(index.ispec, index.iz, index.iy, index.ix, 1), mask,
        tag_type());
    boundary.face_normal(2) = Kokkos::Experimental::simd_partial_load(
        &face_normal(index.ispec, index.iz, index.iy, index.ix, 2), mask,
        tag_type());
    boundary.face_weight = Kokkos::Experimental::simd_partial_load(
        &face_weight(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == true,
                int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const IndexType &index,
                 specfem::point::boundary<
                     specfem::element::boundary_tag::composite_stacey_dirichlet,
                     dimension_tag, true> &boundary) const {
    using simd = typename specfem::datatype::simd<type_real, true>;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const auto mask = index.template get_mask<simd>();

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }

    boundary.face_normal(0) = Kokkos::Experimental::simd_partial_load(
        &face_normal(index.ispec, index.iz, index.iy, index.ix, 0), mask,
        tag_type());
    boundary.face_normal(1) = Kokkos::Experimental::simd_partial_load(
        &face_normal(index.ispec, index.iz, index.iy, index.ix, 1), mask,
        tag_type());
    boundary.face_normal(2) = Kokkos::Experimental::simd_partial_load(
        &face_normal(index.ispec, index.iz, index.iy, index.ix, 2), mask,
        tag_type());
    boundary.face_weight = Kokkos::Experimental::simd_partial_load(
        &face_weight(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
  }

  // ── Host load methods ────────────────────────────────────────────────────

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == false,
                int> = 0>
  inline void load_on_host(const IndexType &index,
                           specfem::point::boundary<boundary_tag, dimension_tag,
                                                    false> &boundary) const {
    boundary.tag += h_quadrature_point_boundary_tag(index.ispec, index.iz,
                                                    index.iy, index.ix);
    boundary.face_normal(0) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 0);
    boundary.face_normal(1) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 1);
    boundary.face_normal(2) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 2);
    boundary.face_weight =
        h_face_weight(index.ispec, index.iz, index.iy, index.ix);
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == false,
                int> = 0>
  inline void
  load_on_host(const IndexType &index,
               specfem::point::boundary<
                   specfem::element::boundary_tag::composite_stacey_dirichlet,
                   dimension_tag, false> &boundary) const {
    boundary.tag += h_quadrature_point_boundary_tag(index.ispec, index.iz,
                                                    index.iy, index.ix);
    boundary.face_normal(0) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 0);
    boundary.face_normal(1) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 1);
    boundary.face_normal(2) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 2);
    boundary.face_weight =
        h_face_weight(index.ispec, index.iz, index.iy, index.ix);
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == true,
                int> = 0>
  inline void load_on_host(const IndexType &index,
                           specfem::point::boundary<boundary_tag, dimension_tag,
                                                    true> &boundary) const {
    using simd = typename specfem::datatype::simd<type_real, true>;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const auto mask = index.template get_mask<simd>();

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }

    boundary.face_normal(0) = Kokkos::Experimental::simd_partial_load(
        &h_face_normal(index.ispec, index.iz, index.iy, index.ix, 0), mask,
        tag_type());
    boundary.face_normal(1) = Kokkos::Experimental::simd_partial_load(
        &h_face_normal(index.ispec, index.iz, index.iy, index.ix, 1), mask,
        tag_type());
    boundary.face_normal(2) = Kokkos::Experimental::simd_partial_load(
        &h_face_normal(index.ispec, index.iz, index.iy, index.ix, 2), mask,
        tag_type());
    boundary.face_weight = Kokkos::Experimental::simd_partial_load(
        &h_face_weight(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == true,
                int> = 0>
  inline void
  load_on_host(const IndexType &index,
               specfem::point::boundary<
                   specfem::element::boundary_tag::composite_stacey_dirichlet,
                   dimension_tag, true> &boundary) const {
    using simd = typename specfem::datatype::simd<type_real, true>;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const auto mask = index.template get_mask<simd>();

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }

    boundary.face_normal(0) = Kokkos::Experimental::simd_partial_load(
        &h_face_normal(index.ispec, index.iz, index.iy, index.ix, 0), mask,
        tag_type());
    boundary.face_normal(1) = Kokkos::Experimental::simd_partial_load(
        &h_face_normal(index.ispec, index.iz, index.iy, index.ix, 1), mask,
        tag_type());
    boundary.face_normal(2) = Kokkos::Experimental::simd_partial_load(
        &h_face_normal(index.ispec, index.iz, index.iy, index.ix, 2), mask,
        tag_type());
    boundary.face_weight = Kokkos::Experimental::simd_partial_load(
        &h_face_weight(index.ispec, index.iz, index.iy, index.ix), mask,
        tag_type());
  }
};

} // namespace specfem::assembly::boundaries_impl
