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
 * @brief Data container for acoustic free surface boundary conditions in 3D
 *
 * Stores per-quadrature-point boundary tag data for acoustic elements whose
 * top face (face_type::top) is designated as a free surface boundary. The free
 * surface condition enforces zero pressure (traction) at the boundary.
 *
 */
template <>
struct acoustic_free_surface<specfem::element::dimension_tag::dim3> {
private:
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::acoustic_free_surface;

public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  /**
   * @brief Kokkos view for boundary tags per quadrature point on device
   *
   * Dimensions: [nspec_acoustic_free_surface, ngllz, nglly, ngllx]
   */
  using BoundaryTagView =
      Kokkos::View<specfem::element::boundary_tag_container ****,
                   Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  BoundaryTagView quadrature_point_boundary_tag; ///< Device boundary tags
  BoundaryTagView::host_mirror_type h_quadrature_point_boundary_tag; ///< Host
                                                                     ///< mirror

  acoustic_free_surface() = default;

  /**
   * @brief Construct from mesh boundary information
   *
   * @param nspec Total number of spectral elements
   * @param ngllz GLL points in z direction
   * @param nglly GLL points in y direction
   * @param ngllx GLL points in x direction
   * @param mesh Mesh boundary data (all domain boundary faces)
   * @param mesh_assembly Assembly mesh with coordinate and mapping info
   * @param boundary_index_mapping Mapping from compute element index to local
   *        acoustic_free_surface index (-1 if element has no free surface)
   * @param boundary_tag Per-element boundary tag containers to update
   */
  acoustic_free_surface(
      const int nspec, const int ngllz, const int nglly, const int ngllx,
      const specfem::mesh::mesh<dimension_tag> &mesh,
      const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
      const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
      std::vector<specfem::element::boundary_tag_container> &boundary_tag);

  // ── Boundary tag accessors ───────────────────────────────────────────────

  /**
   * @brief Boundary tag container at a quadrature point of an acoustic free
   * surface element (device)
   *
   * @param ispec Local acoustic free surface element index (see
   *        specfem::assembly::boundaries::acoustic_free_surface_index_mapping)
   * @param iz GLL point index in z direction
   * @param iy GLL point index in y direction
   * @param ix GLL point index in x direction
   * @return Boundary tag container at the quadrature point
   */
  KOKKOS_FORCEINLINE_FUNCTION specfem::element::boundary_tag_container
  get_boundary_tag_on_device(const int ispec, const int iz, const int iy,
                             const int ix) const {
    return quadrature_point_boundary_tag(ispec, iz, iy, ix);
  }

  /**
   * @brief Boundary tag container at a quadrature point of an acoustic free
   * surface element (host)
   *
   * @param ispec Local acoustic free surface element index (see
   *        specfem::assembly::boundaries::h_acoustic_free_surface_index_mapping)
   * @param iz GLL point index in z direction
   * @param iy GLL point index in y direction
   * @param ix GLL point index in x direction
   * @return Boundary tag container at the quadrature point
   */
  inline specfem::element::boundary_tag_container
  get_boundary_tag_on_host(const int ispec, const int iz, const int iy,
                           const int ix) const {
    return h_quadrature_point_boundary_tag(ispec, iz, iy, ix);
  }

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

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }
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

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }
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

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }
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

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }
  }
};

} // namespace specfem::assembly::boundaries_impl
