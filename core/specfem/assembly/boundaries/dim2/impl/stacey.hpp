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
 * @brief Stacey absorbing boundary condition implementation for 2D SEM
 * simulations
 *
 * This class provides the implementation for Stacey absorbing boundary
 * conditions in 2D spectral element simulations. Stacey boundary conditions
 * work by applying a velocity-proportional damping term at boundary edges that
 * approximates the impedance of the outgoing waves. The boundary condition is
 * implemented as:
 *
 * \f$ \mathbf{T} \cdot \mathbf{n} = -\rho c \mathbf{v} \cdot \mathbf{n} \f$
 *
 * where \f$\mathbf{T}\f$ is the traction, \f$\mathbf{n}\f$ is the outward
 * normal,
 * \f$\rho\f$ is density, \f$c\f$ is wave speed, and \f$\mathbf{v}\f$ is
 * velocity.
 *
 * **Applications:**
 * - Seismic wave propagation in unbounded domains
 * - Reduction of spurious reflections from domain boundaries
 * - Truncation of infinite geological media
 * - Wave scattering problems requiring non-reflecting boundaries
 *
 * **Implementation Details:**
 * The class stores geometric and physical data required for Stacey boundary
 * implementation:
 * - Boundary tags for identification during assembly
 * - Edge normal vectors for computing outward flux
 * - Edge integration weights for boundary integral evaluation
 * - Both device and host storage for hybrid CPU/GPU execution
 *
 */
template <> struct stacey<specfem::dimension::type::dim2> {
private:
  /**
   * @name Private Constants
   */
  ///@{
  /**
   * @brief Static boundary tag identifier for Stacey absorbing boundary
   * conditions
   */
  constexpr static auto boundary_tag = specfem::element::boundary_tag::stacey;
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
   * Stores boundary tag information for quadrature points within Stacey
   * boundary elements. Dimensions: [nspec_stacey, ngllz, ngllx]
   */
  using BoundaryTagView =
      Kokkos::View<specfem::element::boundary_tag_container ***,
                   Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing edge normal vectors on device
   *
   * Stores outward normal vectors at boundary edges for each quadrature point.
   * Dimensions: [nspec_stacey, ngllz, ngllx, ndim] where ndim=2 for 2D problems
   */
  using EdgeNormalView = Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                      Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing edge integration weights on device
   *
   * Stores integration weights for boundary edge quadrature.
   * Dimensions: [nspec_stacey, ngllz, ngllx]
   */
  using EdgeWeightView = Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                      Kokkos::DefaultExecutionSpace>;
  ///@}

  /**
   * @name Data Members
   */
  ///@{
  /**
   * @brief Device-accessible boundary tag information for quadrature points
   *
   * Contains boundary tag containers for every quadrature point within spectral
   * elements that have Stacey absorbing boundaries. Used during device kernel
   * execution to identify points requiring Stacey boundary treatment.
   *
   * **Dimensions:** [nspec_stacey, ngllz, ngllx]
   * - nspec_stacey: Number of elements with Stacey boundaries
   * - ngllz, ngllx: Number of GLL points in z and x directions
   */
  BoundaryTagView quadrature_point_boundary_tag;

  /**
   * @brief Host-accessible mirror of boundary tag information
   *
   * Host mirror of quadrature_point_boundary_tag for CPU access during
   * initialization, debugging, and host-side computations.
   */
  BoundaryTagView::HostMirror h_quadrature_point_boundary_tag;

  /**
   * @brief Device-accessible edge normal vectors for boundary quadrature points
   *
   * Stores the outward unit normal vectors at boundary edges for each
   * quadrature point. These normals are essential for computing the
   * impedance-based absorption terms in the Stacey boundary condition
   * formulation.
   *
   * **Dimensions:** [nspec_stacey, ngllz, ngllx, 2]
   * - Component 0: Normal vector x-component
   * - Component 1: Normal vector z-component
   *
   */
  EdgeNormalView edge_normal;

  /**
   * @brief Device-accessible edge integration weights for boundary quadrature
   *
   * Contains integration weights for evaluating boundary integrals along Stacey
   * edges. These weights incorporate the Jacobian transformation from reference
   * to physical coordinates and are zero for interior points not on boundary
   * edges.
   *
   * **Dimensions:** [nspec_stacey, ngllz, ngllx]
   *
   */
  EdgeWeightView edge_weight;

  /**
   * @brief Host-accessible mirror of edge normal vectors
   *
   * Host mirror of edge_normal for CPU access and initialization.
   */
  EdgeNormalView::HostMirror h_edge_normal;

  /**
   * @brief Host-accessible mirror of edge integration weights
   *
   * Host mirror of edge_weight for CPU access and initialization.
   */
  EdgeWeightView::HostMirror h_edge_weight;
  ///@}

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Default constructor
   *
   * Creates an empty Stacey boundary condition container with uninitialized
   * data members. Data must be set up via assignment or the parametrized
   * constructor.
   */
  stacey() = default;

  /**
   * @brief Construct Stacey absorbing boundary condition data from mesh
   * information
   *
   * This constructor processes mesher-supplied absorbing boundary information
   * and converts it into per-quadrature-point data required for implementing
   * Stacey absorbing boundary conditions during SEM simulations.
   *
   *
   * @param nspec Number of spectral elements with Stacey absorbing boundaries
   * @param ngllz Number of GLL points in the z (vertical) direction
   * @param ngllx Number of GLL points in the x (horizontal) direction
   * @param stacey Mesh-level absorbing boundary information containing edge
   * lists, normal vectors, and geometric data from the mesher
   * @param mesh Assembly mesh with coordinate information and element
   * connectivity
   * @param jacobian_matrix Jacobian matrix container with basis function
   * derivatives used for computing geometric transformations at quadrature
   * points
   * @param boundary_index_mapping Mapping from global spectral element indices
   * to boundary-local indices for elements with Stacey boundaries
   * @param boundary_tag Vector of boundary tag containers updated with
   * element-level boundary information for assembly processes
   *
   */
  stacey(
      const int nspec, const int ngllz, const int ngllx,
      const specfem::mesh::absorbing_boundary<dimension_tag> &stacey,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
      std::vector<specfem::element::boundary_tag_container> &boundary_tag);
  ///@}

  /**
   * @name Device Data Access Methods
   *
   * Methods for loading Stacey boundary condition data on GPU devices. These
   * functions are optimized for device execution and provide access to
   * geometric and physical data required for absorbing boundary condition
   * implementation.
   */
  ///@{
  /**
   * @brief Load Stacey boundary data for a non-SIMD quadrature point on device
   *
   * This function loads complete Stacey boundary information for a specific
   * quadrature point during GPU kernel execution. It populates the boundary
   * object with all data needed for absorbing boundary condition calculations,
   * including geometric normals and integration weights.
   *
   * @tparam IndexType Must be a valid non-SIMD index type
   * (specfem::point::index)
   *
   * @param index Quadrature point location specifier containing
   * @param boundary Output boundary object populated with Stacey boundary data
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

    boundary.edge_normal(0) = edge_normal(index.ispec, index.iz, index.ix, 0);
    boundary.edge_normal(1) = edge_normal(index.ispec, index.iz, index.ix, 1);
    boundary.edge_weight = edge_weight(index.ispec, index.iz, index.ix);

    return;
  }

  /**
   * @brief Load Stacey boundary data for composite boundary conditions on
   * device
   *
   *
   * @tparam IndexType Must be a valid non-SIMD index type
   * @param index Quadrature point location specifier
   * @param boundary Output composite boundary object populated with Stacey
   * contributions
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
                 specfem::point::boundary<
                     specfem::element::boundary_tag::composite_stacey_dirichlet,
                     dimension_tag, false> &boundary) const {

    boundary.tag +=
        quadrature_point_boundary_tag(index.ispec, index.iz, index.ix);

    boundary.edge_normal(0) = edge_normal(index.ispec, index.iz, index.ix, 0);
    boundary.edge_normal(1) = edge_normal(index.ispec, index.iz, index.ix, 1);
    boundary.edge_weight = edge_weight(index.ispec, index.iz, index.ix);

    return;
  }

  /**
   * @brief Load Stacey boundary data for SIMD quadrature points on device
   *
   *
   * @tparam IndexType Must be a valid SIMD index type
   *
   * @param index SIMD index containing multiple quadrature point locations
   * @param boundary Output SIMD boundary object with vectorized data storage
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

    Kokkos::Experimental::where(mask, boundary.edge_normal(0))
        .copy_from(&edge_normal(index.ispec, index.iz, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_normal(1))
        .copy_from(&edge_normal(index.ispec, index.iz, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_weight)
        .copy_from(&edge_weight(index.ispec, index.iz, index.ix), tag_type());
  }

  /**
   * @brief Load Stacey boundary data for SIMD composite boundaries on device
   *
   * @tparam IndexType Must be a valid SIMD index type
   * @param index SIMD index with masked lanes for multiple quadrature points
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

    Kokkos::Experimental::where(mask, boundary.edge_normal(0))
        .copy_from(&edge_normal(index.ispec, index.iz, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_normal(1))
        .copy_from(&edge_normal(index.ispec, index.iz, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_weight)
        .copy_from(&edge_weight(index.ispec, index.iz, index.ix), tag_type());
  }
  ///@}

  /**
   * @name Host Data Access Methods
   *
   * Methods for loading Stacey boundary condition data on CPU hosts. These
   * functions provide equivalent functionality to device methods but operate on
   * host-accessible data for CPU-based computations, debugging, and
   * initialization.
   */
  ///@{
  /**
   * @brief Load Stacey boundary data for a quadrature point on host
   *
   * @tparam IndexType Must be a valid non-SIMD index type
   * (specfem::point::index)
   *
   * @param index Quadrature point location specifier
   * @param boundary Output boundary object populated with Stacey boundary data
   *
   * @note This is an implementation detail and is typically called by a
   * higher-level @c load_on_host function
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

    boundary.edge_normal(0) = h_edge_normal(index.ispec, index.iz, index.ix, 0);
    boundary.edge_normal(1) = h_edge_normal(index.ispec, index.iz, index.ix, 1);

    boundary.edge_weight = h_edge_weight(index.ispec, index.iz, index.ix);

    return;
  }

  /**
   * @brief Load Stacey boundary data for composite boundaries on host
   *
   * @tparam IndexType Must be a valid non-SIMD index type
   * @param index Quadrature point location specifier
   * @param boundary Output composite boundary object populated with Stacey
   * contributions
   *
   * @note This is an implementation detail and is typically called by a
   * higher-level @c load_on_host function
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

    boundary.edge_normal(0) = h_edge_normal(index.ispec, index.iz, index.ix, 0);
    boundary.edge_normal(1) = h_edge_normal(index.ispec, index.iz, index.ix, 1);

    boundary.edge_weight = h_edge_weight(index.ispec, index.iz, index.ix);

    return;
  }

  /**
   * @brief Load Stacey boundary data for SIMD quadrature points on host
   *
   * @tparam IndexType Must be a valid SIMD index type
   * @param index SIMD index with masked lanes for multiple quadrature points
   * @param boundary Output SIMD boundary object with vectorized data storage
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
  inline void load_on_host(const IndexType &index,
                           specfem::point::boundary<boundary_tag, dimension_tag,
                                                    true> &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.edge_normal(0))
        .copy_from(&h_edge_normal(index.ispec, index.iz, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_normal(1))
        .copy_from(&h_edge_normal(index.ispec, index.iz, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_weight)
        .copy_from(&h_edge_weight(index.ispec, index.iz, index.ix), tag_type());

    return;
  }

  /**
   * @brief Load Stacey boundary data for SIMD composite boundaries on host
   *
   * @tparam IndexType Must be a valid SIMD index type
   * @param index SIMD index with masked lanes for multiple quadrature points
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

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.edge_normal(0))
        .copy_from(&h_edge_normal(index.ispec, index.iz, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_normal(1))
        .copy_from(&h_edge_normal(index.ispec, index.iz, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.edge_weight)
        .copy_from(&h_edge_weight(index.ispec, index.iz, index.ix), tag_type());

    return;
  }
  ///@}
};

} // namespace specfem::assembly::boundaries_impl
