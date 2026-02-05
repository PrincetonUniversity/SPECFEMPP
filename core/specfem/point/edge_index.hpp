#pragma once
#include "enumerations/interface_tags.hpp"

namespace specfem::point {

/**
 * @struct edge_index
 * @brief Primary template for edge index in spectral element meshes
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag> struct edge_index;

/**
 * @brief 2D edge index for spectral element edge access
 *
 * Provides indexing information to locate and access data at specific
 * points along edges in 2D spectral element meshes. Contains both
 * element-level indices (element, edge) and local coordinate indices
 * within the edge.
 */
template <>
struct edge_index<specfem::element::dimension_tag::dim2>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::edge_index,
          specfem::element::dimension_tag::dim2, false> {

  /**
   * @brief Dimension tag for 2D specialization.
   */
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;

  /**
   * @brief Spectral element index.
   */
  int ispec;

  /**
   * @brief Local edge index within the iterator.
   */
  int iedge;

  /**
   * @brief Point index along the edge.
   */
  int ipoint;

  /**
   * @brief Local z-coordinate index within element.
   */
  int iz;

  /**
   * @brief Local x-coordinate index within element.
   */
  int ix;

  /**
   * @brief Mesh entity type for the edge.
   */
  specfem::mesh_entity::dim2::type edge_type;

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  edge_index() = default;

  /**
   * @brief Constructs edge index with all indices.
   *
   * @param ispec_ Element index.
   * @param iedge_ Edge index (0-3 for 2D quad elements).
   * @param ipoint_ Point index along edge.
   * @param iz_ Local z-coordinate index.
   * @param ix_ Local x-coordinate index.
   * @param edge_type_ Mesh entity type for the edge.
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
struct edge_index<specfem::element::dimension_tag::dim3>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::edge_index,
          specfem::element::dimension_tag::dim3, false> {

  /**
   * @brief Dimension tag for 3D specialization.
   */
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;

  /**
   * @brief Spectral element index.
   */
  int ispec;

  /**
   * @brief Local edge index within the iterator.
   */
  int iedge;

  /**
   * @brief Point index along the edge.
   */
  int ipoint;

  /**
   * @brief Local z-coordinate index within element.
   */
  int iz;

  /**
   * @brief Local y-coordinate index within element.
   */
  int iy;

  /**
   * @brief Local x-coordinate index within element.
   */
  int ix;

  /**
   * @brief Mesh entity type for the edge.
   */
  specfem::mesh_entity::dim3::type edge_type;

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  edge_index() = default;

  /**
   * @brief Constructs edge index with all indices.
   *
   * @param ispec_ Element index.
   * @param iedge_ Edge index (0-11 for 3D hexahedral elements).
   * @param ipoint_ Point index along edge.
   * @param iz_ Local z-coordinate index.
   * @param iy_ Local y-coordinate index.
   * @param ix_ Local x-coordinate index.
   * @param edge_type_ Mesh entity type for the edge.
   */
  KOKKOS_INLINE_FUNCTION
  edge_index(const int ispec_, const int iedge_, const int ipoint_,
             const int iz_, const int iy_, const int ix_,
             const specfem::mesh_entity::dim3::type edge_type_)
      : ispec(ispec_), iedge(iedge_), ipoint(ipoint_), iz(iz_), iy(iy_),
        ix(ix_), edge_type(edge_type_) {}
};

} // namespace specfem::point
