#pragma once
#include "enumerations/interface.hpp"

namespace specfem::point {

/**
 * @brief Edge-based indexing system for spectral element boundary and interface
 * operations.
 *
 * The edge_index class provides a comprehensive indexing system for accessing
 * and manipulating data along edges of spectral elements. This is essential for
 * operations that occur on element boundaries, such as boundary condition
 * application, interface coupling, flux computations, and surface integration.
 *
 * **Mathematical Context:**
 * Edge indices enable the evaluation of boundary integrals:
 * \f$
 *   \int_{\Gamma} f(\mathbf{s}) \, d\Gamma = \sum_{e \in \text{edges}}
 * \sum_{i=0}^{N} f(\mathbf{s}_i^{(e)}) w_i^{(e)} J^{(e)}
 * \f$
 * where \f$\Gamma\f$ is the boundary, \f$e\f$ are edge indices, \f$i\f$ are
 * points along edges,
 * \f$w_i^{(e)}\f$ are 1D quadrature weights, and \f$J^{(e)}\f$ is the edge
 * Jacobian.
 *
 * @tparam DimensionTag Spatial dimension determining edge structure:
 *                      - `dim2`: Edges are 1D curves (line segments)
 *                      - `dim3`: Edges are 1D curves on 2D faces
 *
 * @note Edge indexing requires careful handling of orientation and
 *       coordinate mappings to maintain consistency across elements.
 *
 * @see specfem::boundary_conditions for edge-based BC implementations
 * @see specfem::interface for edge-based coupling operations
 *
 * @code
 * // Example: Traversing all points on a 2D element edge
 * using EdgeIndex2D =
 * specfem::point::edge_index<specfem::dimension::type::dim2>;
 *
 * EdgeIndex2D edge_idx;
 * edge_idx.ispec = element_id;
 * edge_idx.iedge = boundary_edge_id;  // 0-3 for quad edges
 *
 * // Process all quadrature points along the edge
 * for (int ipoint = 0; ipoint < NGLL_EDGE; ++ipoint) {
 *   edge_idx.ipoint = ipoint;
 *
 *   // Map edge point to element coordinates
 *   auto [iz, ix] = map_edge_to_element_coords(edge_idx.iedge,
 * edge_idx.ipoint); edge_idx.iz = iz; edge_idx.ix = ix;
 *
 *   // Apply boundary condition
 *   apply_boundary_condition(edge_idx, boundary_value);
 * }
 * @endcode
 */
template <specfem::dimension::type DimensionTag> struct edge_index;

/**
 * @brief 2D edge indexing for boundary operations on quadrilateral spectral
 * elements.
 *
 * This specialization provides comprehensive indexing for accessing quadrature
 * points along the edges of 2D spectral elements. In 2D, element edges are line
 * segments that form the boundary of quadrilateral elements, and this class
 * enables efficient access to data and operations along these 1D boundaries.
 *
 * **2D Edge Structure:**
 * For quadrilateral elements, there are 4 edges with standard numbering:
 * - Edge 0: Bottom edge (iz=0, ix varies)
 * - Edge 1: Right edge (ix=NGLL-1, iz varies)
 * - Edge 2: Top edge (iz=NGLL-1, ix varies)
 * - Edge 3: Left edge (ix=0, iz varies)
 *
 * **Edge Point Mapping:**
 * Each edge contains NGLL quadrature points, and the mapping from edge-local
 * coordinates to element-local coordinates follows specific patterns:
 * \f$
 *   \text{(edge\_point)} \rightarrow \text{(element\_iz, element\_ix)}
 * \f$
 *
 * **Typical Applications:**
 * - Free surface boundary conditions on topographic boundaries
 * - Absorbing boundary conditions for wave simulations
 * - Interface coupling between fluid and solid elements
 * - Flux computations for discontinuous Galerkin methods
 * - Communication patterns in domain decomposition
 *
 * @note The edge orientation and point ordering must be consistent with
 *       the global mesh connectivity to ensure proper data exchange
 *       between neighboring elements.
 *
 * @see specfem::mesh_entity::dim2::type for edge type enumeration
 * @see specfem::boundary_conditions for edge-based boundary implementations
 *
 * @code
 * // Example: Applying free surface BC on top edge of 2D element
 * specfem::point::edge_index<specfem::dimension::type::dim2> edge_pt;
 *
 * edge_pt.ispec = surface_element_id;
 * edge_pt.iedge = 2;  // Top edge
 * edge_pt.edge_type = specfem::mesh_entity::dim2::type::top;
 *
 * // Apply zero-stress condition along entire top edge
 * for (int ipoint = 0; ipoint < NGLL; ++ipoint) {
 *   edge_pt.ipoint = ipoint;
 *   edge_pt.iz = NGLL - 1;  // Top edge
 *   edge_pt.ix = ipoint;    // Varies along edge
 *
 *   // Zero normal stress: \sigma_zz = 0
 *   stress_field(edge_pt.ispec, edge_pt.iz, edge_pt.ix, 1, 1) = 0.0;
 *
 *   // Zero shear stress: \sigma_xz = 0
 *   stress_field(edge_pt.ispec, edge_pt.iz, edge_pt.ix, 0, 1) = 0.0;
 * }
 * @endcode
 */
template <>
struct edge_index<specfem::dimension::type::dim2>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::edge_index,
          specfem::dimension::type::dim2, false> {

  /** @brief Dimension tag for 2D specialization */
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;

  /** @brief Spectral element index */
  int ispec;
  /** @brief Local edge index within the iterator */
  int iedge;
  /** @brief Point index along the edge */
  int ipoint;
  /** @brief Local z-coordinate index within element */
  int iz;
  /** @brief Local x-coordinate index within element */
  int ix;
  /** @brief Mesh entity type for the edge */
  specfem::mesh_entity::dim2::type edge_type;

  /** @brief Default constructor */
  KOKKOS_INLINE_FUNCTION
  edge_index() = default;

  /**
   * @brief Constructs edge index with all indices
   *
   * @param ispec_ Element index
   * @param iedge_ Edge index (0-3 for 2D quad elements)
   * @param ipoint_ Point index along edge
   * @param iz_ Local z-coordinate index
   * @param ix_ Local x-coordinate index
   */
  KOKKOS_INLINE_FUNCTION
  edge_index(const int ispec_, const int iedge_, const int ipoint_,
             const int iz_, const int ix_,
             const specfem::mesh_entity::dim2::type edge_type_)
      : ispec(ispec_), iedge(iedge_), ipoint(ipoint_), iz(iz_), ix(ix_),
        edge_type(edge_type_) {}
};

/**
 * @brief 3D edge index for spectral element edge access
 *
 * Provides indexing information to locate and access data at specific
 * points along edges in 3D spectral element meshes. Contains both
 * element-level indices (element, edge) and local coordinate indices
 * within the edge.
 */
template <>
struct edge_index<specfem::dimension::type::dim3>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::edge_index,
          specfem::dimension::type::dim3, false> {

  /** @brief Dimension tag for 3D specialization */
  static constexpr auto dimension_tag = specfem::dimension::type::dim3;

  /** @brief Spectral element index */
  int ispec;
  /** @brief Local edge index within the iterator */
  int iedge;
  /** @brief Point index along the edge */
  int ipoint;
  /** @brief Local z-coordinate index within element */
  int iz;
  /** @brief Local y-coordinate index within element */
  int iy;
  /** @brief Local x-coordinate index within element */
  int ix;
  /** @brief Mesh entity type for the edge */
  specfem::mesh_entity::dim3::type edge_type;

  /** @brief Default constructor */
  KOKKOS_INLINE_FUNCTION
  edge_index() = default;

  /**
   * @brief Constructs edge index with all indices
   *
   * @param ispec_ Element index
   * @param iedge_ Edge index (0-3 for 2D quad elements)
   * @param ipoint_ Point index along edge
   * @param iz_ Local z-coordinate index
   * @param iy_ Local y-coordinate index
   * @param ix_ Local x-coordinate index
   */
  KOKKOS_INLINE_FUNCTION
  edge_index(const int ispec_, const int iedge_, const int ipoint_,
             const int iz_, const int iy_, const int ix_,
             const specfem::mesh_entity::dim3::type edge_type_)
      : ispec(ispec_), iedge(iedge_), ipoint(ipoint_), iz(iz_), iy(iy_),
        ix(ix_), edge_type(edge_type_) {}
};

} // namespace specfem::point
