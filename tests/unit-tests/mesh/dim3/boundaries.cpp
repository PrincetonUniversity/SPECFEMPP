#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"
#include "specfem_setup.hpp"
#include "test_fixture.hpp"

namespace specfem::test_configuration {

using face_direction =
    specfem::mesh::boundaries<specfem::dimension::type::dim3>::FaceDirection;

/**
 * @brief Represents the total number of faces and elements in the mesh.
 *
 */
struct TotalFaces {
  int nfaces;    ///< Total number of faces in the mesh
  int nelements; ///< Total number of elements in the mesh

  TotalFaces(int nfaces, int nx, int ny, int nz)
      : nfaces(nfaces), nelements(nx * ny * nz) {}
};

/**
 * @brief Represents an absorbing boundary face in 3D space.
 *
 */
struct Boundaries3D {
  int element_id; ///< Identifier for the element containing the boundary face
  bool is_boundary_element; ///< Flag indicating if the element is a boundary
                            ///< element
  specfem::mesh_entity::dim3::type face_type; ///< Type of the boundary face
  face_direction direction; ///< Direction of the boundary face

  Boundaries3D(int element_id, specfem::mesh_entity::dim3::type face_type,
               face_direction direction)
      : element_id(element_id), is_boundary_element(true), face_type(face_type),
        direction(direction) {}

  Boundaries3D(int element_id)
      : element_id(element_id), is_boundary_element(false) {}
};

struct ExpectedBoundaries3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3; ///< Dimension of the faces
  TotalFaces total_faces;             ///< Total faces and elements in the mesh
  std::vector<Boundaries3D> faces;    ///< List of expected absorbing boundary
                                      ///< faces

  ExpectedBoundaries3D(TotalFaces total_faces,
                       const std::initializer_list<Boundaries3D> faces)
      : total_faces(total_faces), faces(faces) {}

  void
  check(const specfem::mesh::boundaries<dimension> &absorbing_boundary) const {
    // Verify that the absorbing boundary object has the expected number of
    // faces
    if (absorbing_boundary.nfaces != total_faces.nfaces) {
      FAIL() << "Total number of absorbing boundary faces mismatch. "
             << "Expected: " << total_faces.nfaces << ", "
             << "Got: " << absorbing_boundary.nfaces << std::endl;
    }

    // Verify that the absorbing boundary object has the expected number of
    // elements
    if (absorbing_boundary.nspec != total_faces.nelements) {
      FAIL() << "Total number of elements mismatch. "
             << "Expected: " << total_faces.nelements << ", "
             << "Got: " << absorbing_boundary.nspec << std::endl;
    }
    // Check each expected absorbing boundary face
    for (const auto &expected_face : faces) {
      if (expected_face.is_boundary_element) {
        bool found = false;
        for (int i = 0; i < absorbing_boundary.nfaces; ++i) {
          if (absorbing_boundary.index_mapping(i) == expected_face.element_id &&
              absorbing_boundary.face_type(i) == expected_face.face_type &&
              absorbing_boundary.face_direction(i) == expected_face.direction) {
            found = true;
            break;
          }
        }
        if (!found) {
          FAIL() << "Absorbing boundary face not found for element "
                 << expected_face.element_id << " with face type "
                 << specfem::mesh_entity::dim3::to_string(
                        expected_face.face_type)
                 << "." << std::endl;
        }
      } else {
        bool found = false;
        for (int i = 0; i < absorbing_boundary.nfaces; ++i) {
          if (absorbing_boundary.index_mapping(i) == expected_face.element_id) {
            found = true;
            break;
          }
        }
        if (found) {
          FAIL() << "Non-absorbing element " << expected_face.element_id
                 << " incorrectly marked as absorbing." << std::endl;
        }
      }
    }
    SUCCEED()
        << "All expected absorbing boundary faces are present and correct."
        << std::endl;
  }
};

} // namespace specfem::test_configuration

using namespace specfem::test_configuration;

static const std::unordered_map<std::string, ExpectedBoundaries3D>
    expected_absorbing_boundary_faces_map = {
      { "EightNodeElastic",
        ExpectedBoundaries3D(
            TotalFaces(24, 2, 2, 2),
            {
                // X_MIN boundary face
                Boundaries3D(0, specfem::mesh_entity::dim3::type::left,
                             face_direction::X_MIN),
                // X_MAX boundary face
                Boundaries3D(1, specfem::mesh_entity::dim3::type::right,
                             face_direction::X_MAX),
                // Y_MIN boundary face
                Boundaries3D(0, specfem::mesh_entity::dim3::type::front,
                             face_direction::Y_MIN),
                // Y_MAX boundary face
                Boundaries3D(2, specfem::mesh_entity::dim3::type::back,
                             face_direction::Y_MAX),
                // Z_MIN boundary face
                Boundaries3D(0, specfem::mesh_entity::dim3::type::bottom,
                             face_direction::Z_MIN),
                // Z_MAX boundary face
                Boundaries3D(4, specfem::mesh_entity::dim3::type::top,
                             face_direction::Z_MAX),
            }) }
    };

TEST_P(Mesh3DTest, Boundaries) {
  const auto &param_name = GetParam();
  if (expected_absorbing_boundary_faces_map.find(param_name) ==
      expected_absorbing_boundary_faces_map.end()) {
    GTEST_SKIP() << "No ground truth defined for test case: " << param_name
                 << std::endl;
    return;
  }

  const auto &mesh = getMesh();
  const auto &absorbing_boundaries = mesh.boundaries;
  const auto &expected = expected_absorbing_boundary_faces_map.at(param_name);
  expected.check(absorbing_boundaries);
}
