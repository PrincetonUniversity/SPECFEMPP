#pragma once

namespace specfem::point {

/**
 * @brief Primary template for edge index in spectral element meshes
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::dimension::type DimensionTag> struct edge_index;

/**
 * @brief 2D edge index for spectral element edge access
 *
 * Provides indexing information to locate and access data at specific
 * points along edges in 2D spectral element meshes. Contains both
 * element-level indices (element, edge) and local coordinate indices
 * within the edge.
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
             const int iz_, const int ix_)
      : ispec(ispec_), iedge(iedge_), ipoint(ipoint_), iz(iz_), ix(ix_) {}
};

} // namespace specfem::point
