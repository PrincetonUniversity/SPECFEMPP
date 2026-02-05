#pragma once
#include "specfem/element.hpp"
#include "specfem/mesh_entity.hpp"

namespace specfem::point {

/**
 * @struct face_index
 * @brief Primary template for face index in 3D spectral element meshes
 * @tparam DimensionTag Spatial dimension (dim3 only)
 */
template <specfem::element::dimension_tag DimensionTag> struct face_index;

/**
 * @brief 3D face index for spectral element face access
 *
 * Provides indexing information to locate and access data at specific
 * points on faces in 3D spectral element meshes. Contains both
 * element-level indices (element, face) and local coordinate indices
 * within the face (2D surface with ngll x ngll points).
 */
template <>
struct face_index<specfem::element::dimension_tag::dim3>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::face_index,
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
   * @brief Local face index within the iterator.
   */
  int iface;

  /**
   * @brief First point index on the face (local i-coordinate).
   */
  int ipoint_i;

  /**
   * @brief Second point index on the face (local j-coordinate).
   */
  int ipoint_j;

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
   * @brief Mesh entity type for the face.
   */
  specfem::mesh_entity::dim3::type face_type;

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  face_index() = default;

  /**
   * @brief Constructs face index with all indices.
   *
   * @param ispec_ Element index.
   * @param iface_ Face index (0-5 for 3D hexahedral elements).
   * @param ipoint_i_ First point index on face.
   * @param ipoint_j_ Second point index on face.
   * @param iz_ Local z-coordinate index.
   * @param iy_ Local y-coordinate index.
   * @param ix_ Local x-coordinate index.
   * @param face_type_ Mesh entity type for the face.
   */
  KOKKOS_INLINE_FUNCTION
  face_index(const int ispec_, const int iface_, const int ipoint_i_,
             const int ipoint_j_, const int iz_, const int iy_, const int ix_,
             const specfem::mesh_entity::dim3::type face_type_)
      : ispec(ispec_), iface(iface_), ipoint_i(ipoint_i_), ipoint_j(ipoint_j_),
        iz(iz_), iy(iy_), ix(ix_), face_type(face_type_) {}
};

} // namespace specfem::point
