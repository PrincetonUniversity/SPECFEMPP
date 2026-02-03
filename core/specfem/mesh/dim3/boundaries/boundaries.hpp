/**
 * @file boundaries.hpp
 * @brief Domain boundary management for 3D spectral element meshes
 *
 * This file contains the boundaries template specialization for 3D meshes.
 * The class manages all boundary face information read from MESHFEM3D
 * database files, including absorbing boundaries that implement Stacey
 * absorbing boundary conditions, as well as other domain boundaries such
 * as free surfaces and interface boundaries.
 */

#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::mesh {

template <specfem::dimension::type DimensionTag> struct Boundaries;

/**
 * @brief Domain boundary manager for 3D spectral element meshes
 *
 * This template specialization manages all boundary face information for 3D
 * spectral element simulations, including but not limited to absorbing
 * boundaries. The class stores comprehensive boundary data read from MESHFEM3D
 * database files, encompassing:
 *
 * - **Absorbing boundaries**: Stacey conditions to prevent spurious reflections
 * - **Free surface boundaries**: Traction-free conditions at domain surfaces
 * - **Interface boundaries**: Coupling conditions between different media
 * - **Other domain boundaries**: Any faces at the mesh exterior
 *
 * This unified boundary representation allows for efficient processing of all
 * boundary conditions in the spectral element simulation workflow.
 *
 * @code
 * // Example usage with MESHFEM3D database file
 * std::ifstream database_stream("MESH/boundaries.bin", std::ios::binary);
 * auto control_nodes = read_control_nodes(database_stream);
 *
 * // Read all domain boundaries from database
 * auto boundaries =
 *     specfem::io::mesh_impl::fortran::dim3::meshfem3d::read_boundaries(
 *         database_stream, control_nodes);
 *
 * // Process all boundary faces
 * for (int iface = 0; iface < boundaries.nfaces; ++iface) {
 *     int element_index = boundaries.index_mapping(iface);
 *     auto face_type = boundaries.face_type(iface);
 *
 *     // Apply appropriate boundary condition based on face type
 *     switch (face_type) {
 *         case specfem::mesh_entity::dim3::type::bottom:
 *             // Process bottom boundary
 *             break;
 *         case specfem::mesh_entity::dim3::type::top:
 *             // Process top boundary (e.g., free surface)
 *             break;
 *         // ... handle other face types
 *     }
 * }
 * @endcode
 */
template <> struct boundaries<specfem::dimension::type::dim3> {
private:
  /** @brief Enumeration of face directions for 3D boundaries */
  enum class face_direction : int {
    X_MIN = 1,
    X_MAX = 2,
    Y_MIN = 3,
    Y_MAX = 4,
    Z_MIN = 5,
    Z_MAX = 6
  };

public:
  /** @brief Enumeration type alias for face directions */
  using FaceDirection = face_direction;

private:
  /** @brief Kokkos view type for storing element indices on host memory */
  using IndexViewType = Kokkos::View<int *, Kokkos::HostSpace>;

  /** @brief Kokkos view type for storing face types on host memory */
  using FaceViewType =
      Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>;

  /** @brief Kokkos view type for storing face directions on host memory */
  using FaceDirectionViewType =
      Kokkos::View<face_direction *, Kokkos::HostSpace>;

public:
  /** @brief Dimension tag for template specialization (always dim3) */
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  /**
   * @brief Default constructor
   *
   * Creates an empty boundaries object with zero faces and elements.
   * Use this constructor when boundaries are not required or will be
   * populated later through IO operations or explicit assignment.
   */
  boundaries() = default;

  /**
   * @brief Parameterized constructor for domain boundaries
   *
   * Constructs a boundaries object with the specified number of boundary
   * faces and total spectral elements. Allocates Kokkos host memory views
   * for storing element indices and face types read from MESHFEM3D database
   * files. This constructor handles all types of domain boundaries, not just
   * absorbing boundaries.
   *
   * @param nfaces Number of boundary faces to allocate storage for
   * @param nspec Total number of spectral elements in the mesh
   *
   * @note The constructor initializes Kokkos views with labeled memory for
   * debugging and profiling purposes. The nspec parameter provides mesh
   * context for boundary processing algorithms.
   *
   * @code
   * // Create boundaries container for 120 faces in a mesh with 1000 elements
   * specfem::mesh::boundaries<specfem::dimension::type::dim3>
   *     boundaries(120, 1000);
   *
   * // Views are now allocated and ready for data from MESHFEM3D
   * assert(boundaries.nfaces == 120);
   * assert(boundaries.nspec == 1000);
   * assert(boundaries.index_mapping.extent(0) == 120);
   * assert(boundaries.face_type.extent(0) == 120);
   * @endcode
   */
  boundaries(const int nfaces, const int nspec)
      : nfaces(nfaces), nspec(nspec),
        index_mapping("specfem::mesh::boundaries::index_mapping", nfaces),
        face_type("specfem::mesh::boundaries::face_type", nfaces),
        face_direction("specfem::mesh::boundaries::face_direction", nfaces) {}

  /** @brief Total number of domain boundary faces */
  int nfaces;

  /**
   * @brief Total number of spectral elements in the mesh
   */
  int nspec;

  /**
   * @brief Mapping from boundary face index to spectral element index
   *
   * This Kokkos host view stores the spectral element indices that contain
   * boundary faces. Each entry corresponds to the element that owns the
   * boundary face at the same index position.
   */
  IndexViewType index_mapping;

  /**
   * @brief Type identifier for each domain boundary face
   *
   * This Kokkos host view stores the geometric face type (bottom, top, front,
   * back, left, right) for each boundary face.
   *
   * @see specfem::mesh_entity::dim3::type for complete face enumeration
   */
  FaceViewType face_type;

  /**
   * @brief Direction identifier for each domain boundary face
   *
   * This Kokkos host view stores the directional classification (X_MIN, X_MAX,
   * Y_MIN, Y_MAX, Z_MIN, Z_MAX) for each boundary face.
   *
   * This Kokkos host view is essential for applying boundary conditions
   * in the correct direction during spectral element computations.
   */
  FaceDirectionViewType face_direction;

  /**
   * @brief Filter boundary faces by specified direction
   *
   * Returns Kokkos views containing the element indices and face types for
   * all boundary faces that match the given direction (X_MIN, X_MAX, Y_MIN,
   * Y_MAX, Z_MIN, Z_MAX). This allows for efficient processing of boundary
   * conditions applied to specific domain boundaries.
   *
   * @param dir The face direction to filter by (X_MIN, X_MAX, Y_MIN, Y_MAX,
   * Z_MIN, Z_MAX)
   *
   * @return A tuple of Kokkos views:
   *         - First: View of element indices for matching boundary faces
   *         - Second: View of face types for matching boundary faces
   *
   * @code
   * // Example: Get all faces on the X_MIN boundary
   * auto [indices, faces] = boundaries.filter(
   *     specfem::mesh::boundaries<specfem::dimension::type::dim3>::face_direction::X_MIN);
   *
   * // Process each face on the X_MIN boundary
   * for (size_t i = 0; i < indices.extent(0); ++i) {
   *     int element_index = indices(i);
   *     auto face_type = faces(i);
   *     // Apply X_MIN boundary condition based on face_type
   * }
   * @endcode
   */
  std::tuple<IndexViewType, FaceViewType> filter(enum face_direction dir) {
    std::vector<int> indices;
    std::vector<specfem::mesh_entity::dim3::type> faces;

    for (int i = 0; i < nfaces; ++i) {
      if (face_direction(i) == dir) {
        indices.push_back(index_mapping(i));
        faces.push_back(face_type(i));
      }
    }

    // Create Kokkos views to return
    IndexViewType indices_view("indices_view", indices.size());
    FaceViewType faces_view("faces_view", faces.size());

    for (size_t i = 0; i < indices.size(); ++i) {
      indices_view(i) = indices[i];
      faces_view(i) = faces[i];
    }

    return std::make_tuple(indices_view, faces_view);
  }
};

} // namespace specfem::mesh
