#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "enumerations/interface.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Data container used to store acoustic free surface boundary
 * information
 *
 * This class provides data structures and data access functions required to
 * implement acoustic free surface boundary conditions in 2D spectral element
 * simulations.
 *
 * **Physical Background:**
 * An acoustic free surface boundary condition enforces zero traction (pressure)
 * at the boundary. Mathematically, this corresponds to setting the acceleration
 * to zero at quadrature points located on the free surface. This is physically
 * appropriate for:
 * - Air-water interfaces
 * - Fluid-vacuum boundaries
 * - Ocean-atmosphere interfaces in seismic modeling
 *
 * **Implementation Details:**
 * The class stores boundary tag information for each quadrature point on
 * elements that contain acoustic free surface boundaries. The boundary tags are
 * used during the assembly process to identify which quadrature points require
 * free surface treatment.
 *
 */
template <> struct acoustic_free_surface<specfem::dimension::type::dim2> {
private:
  /**
   * @name Private Constants
   */
  ///@{
  /**
   * @brief Static boundary tag identifier for acoustic free surface conditions
   */
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::acoustic_free_surface;
  ///@}
public:
  /**
   * @name Public Constants
   */
  ///@{
  /**
   * @brief Dimension tag indicating this is a 2D implementation
   */
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;
  ///@}

  /**
   * @name Type Definitions
   */
  ///@{
  /**
   * @brief Kokkos view type for storing boundary tag containers on device
   *
   * This view stores boundary tag information for all quadrature points within
   * elements that contain acoustic free surface boundaries. The view has
   * dimensions [nspec, ngllz, ngllx] where nspec is the number of spectral
   * elements with free surface boundaries.
   */
  using BoundaryTagView =
      Kokkos::View<specfem::element::boundary_tag_container ***,
                   Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  ///@}

  /**
   * @name Data Members
   */
  ///@{
  /**
   * @brief Device-accessible boundary tag information for quadrature points
   *
   * This view contains boundary tag containers for every quadrature point
   * within spectral elements that have acoustic free surface boundaries. The
   * boundary tags are used during device kernel execution to identify which
   * points require free surface treatment.
   *
   * **Dimensions:** [nspec_acoustic_free_surface, ngllz, ngllx]
   * - nspec_acoustic_free_surface: Number of elements with free surface
   * boundaries
   * - ngllz, ngllx: Number of GLL points in z and x directions respectively
   */
  BoundaryTagView quadrature_point_boundary_tag;

  /**
   * @brief Host-accessible mirror of boundary tag information
   *
   * This is the host mirror of `quadrature_point_boundary_tag`, used for CPU
   * access during initialization, debugging, and host-side computations. Data
   * is synchronized between host and device as needed.
   */
  BoundaryTagView::HostMirror h_quadrature_point_boundary_tag;
  ///@}

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Default constructor
   *
   * Creates an empty acoustic free surface boundary condition container.
   * Data members are left uninitialized and must be set up via assignment
   * or the parametrized constructor.
   */
  acoustic_free_surface() = default;

  /**
   * @brief Construct acoustic free surface boundary condition data from mesh
   * information
   *
   * This constructor processes mesher-supplied boundary information and
   * converts it into per-quadrature-point data required for implementing
   * acoustic free surface boundary conditions during SEM simulations.
   *
   * @param nspec Number of spectral elements with acoustic free surface
   * boundaries
   * @param ngllz Number of GLL points in the z (vertical) direction
   * @param ngllx Number of GLL points in the x (horizontal) direction
   * @param acoustic_free_surface Mesh-level acoustic free surface boundary
   * information containing element lists and edge definitions
   * @param mesh Assembly mesh containing coordinate and connectivity
   * information
   * @param boundary_index_mapping Mapping from global spectral element indices
   * to boundary-specific indices for elements with free surface boundaries
   * @param boundary_tag Vector of boundary tag containers that will be
   * populated with per-element boundary information
   *
   */
  acoustic_free_surface(
      const int nspec, const int ngllz, const int ngllx,
      const specfem::mesh::acoustic_free_surface<dimension_tag>
          &acoustic_free_surface,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
      std::vector<specfem::element::boundary_tag_container> &boundary_tag);
  ///@}

  /**
   * @name Device Data Access Methods
   *
   */
  ///@{
  /**
   * @brief Load acoustic free surface boundary data for a non-SIMD quadrature
   * point on device
   *
   * @tparam IndexType Must be a valid index type (specfem::point::index) with
   * SIMD disabled
   *
   * @param index Quadrature point location specifier containing
   * @param boundary Output boundary object
   *
   * @note This is an implementation detail and is typically called by a
   * higher-level @c load_on_device function
   */
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

    boundary.tag +=
        quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);
    return;
  }

  /**
   * @brief Load acoustic free surface data for composite boundary conditions on
   * device
   *
   * @tparam IndexType Must be a valid index type with SIMD disabled
   *
   * @param index Quadrature point location specifier
   * @param boundary Output composite boundary object. The acoustic free surface
   *                 contribution is accumulated into this boundary's tag field.
   *
   * @see This is an implementation detail and is typically called by a
   * higher-level @c load_on_device function
   */
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

    boundary.tag +=
        quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);
    return;
  }

  /**
   * @brief Load acoustic free surface boundary data for SIMD quadrature points
   * on device
   *
   * @tparam IndexType Must be a valid SIMD index type
   *
   * @param index SIMD index containing multiple quadrature point locations
   * @param boundary Output SIMD boundary object with vectorized tag storage
   *
   * @note This is an implementation detail and is typically called by a
   * higher-level @c load_on_device function
   */
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

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(index.ispec + lane,
                                                            index.iz, index.ix);
      }
    }

    return;
  }

  /**
   * @brief Load acoustic free surface data for SIMD composite boundaries on
   * device
   *
   * @tparam IndexType Must be a valid SIMD index type
   * @param index SIMD index with masked lanes
   * @param boundary Output SIMD composite boundary object
   *
   * @note This is an implementation detail and is typically called by a
   * higher-level @c load_on_device function
   */
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

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(index.ispec + lane,
                                                            index.iz, index.ix);
      }
    }

    return;
  }
  ///@}

  /**
   * @name Host Data Access Methods
   *
   * Methods for loading boundary condition data on CPU hosts. These functions
   * provide equivalent functionality to the device methods but operate on
   * host-accessible data.
   */
  ///@{
  /**
   * @brief Load acoustic free surface boundary data for a quadrature point on
   * host
   *
   * @tparam IndexType Must be a valid non-SIMD index type
   * @param index Quadrature point location specifier
   * @param boundary Output boundary object for tag accumulation
   *
   * @note This is an implementation detail and is typically called by a
   * higher-level
   * @c load_on_host function
   */
  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == false,
                int> = 0>
  inline void load_on_host(const IndexType &index,
                           specfem::point::boundary<boundary_tag, dimension_tag,
                                                    false> &boundary) const {
    boundary.tag +=
        h_quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);
    return;
  }

  /**
   * @brief Load acoustic free surface data for composite boundaries on host
   *
   *
   * @tparam IndexType Must be a valid non-SIMD index type
   * @param index Quadrature point location specifier
   * @param boundary Output composite boundary object
   *
   * @note This is an implementation detail and is typically called by a
   * higher-level
   * @c load_on_host function
   */
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

    boundary.tag +=
        h_quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);
    return;
  }

  /**
   * @brief Load acoustic free surface boundary data for SIMD quadrature points
   * on host
   *
   * @tparam IndexType Must be a valid SIMD index type
   * @param index SIMD index with masked lanes for multiple quadrature points
   * @param boundary Output SIMD boundary object with vectorized storage
   *
   * @note This is an implementation detail and is typically called by a
   * higher-level @c load_on_device function
   */
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

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.ix);
      }
    }
    return;
  }

  /**
   * @brief Load acoustic free surface data for SIMD composite boundaries on
   * host
   *
   * @tparam IndexType Must be a valid SIMD index type
   * @param index SIMD index with masked lanes
   * @param boundary Output SIMD composite boundary object
   *
   * @note This is an implementation detail and is typically called by a
   * higher-level @c load_on_host function
   */
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

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.ix);
      }
    }
    return;
  }
  ///@}
};
} // namespace specfem::assembly::boundaries_impl
