
#include <any>
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "enumerations/interface.hpp"
#include "medium/material.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"
#include "specfem_setup.hpp"
#include "test_fixture.hpp"

namespace specfem::test_configuration {

/**
 * @brief Represents the total number of materials and elements in the mesh.
 *
 */
struct TotalMaterials {
  int nmaterials;
  int nelements;

  TotalMaterials(int nmaterials, int nelements)
      : nmaterials(nmaterials), nelements(nelements) {}
};

/**
 * @brief Represents a material associated with a specific element.
 *
 */
struct ElementMaterial {

  specfem::element::medium_tag medium_tag;     ///< Medium type of the material
  specfem::element::property_tag property_tag; ///< Property type of the
                                               ///< material

  int element_id;    ///< ID of the element associated with the material
  std::any material; ///< Material data (type-erased for flexibility)

  template <typename Material>
  ElementMaterial(specfem::element::medium_tag medium_tag,
                  specfem::element::property_tag property_tag, int element_id,
                  const Material &material)
      : medium_tag(medium_tag), property_tag(property_tag),
        element_id(element_id), material(material) {}
};

struct ExpectedMaterials3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  TotalMaterials total_materials; ///< Total materials and elements in the mesh
  std::vector<ElementMaterial> element_materials; ///< List of expected element
                                                  ///< materials

  ExpectedMaterials3D(TotalMaterials total_materials,
                      const std::initializer_list<ElementMaterial> &elements)
      : total_materials(total_materials), element_materials(elements) {}

  void check(const specfem::mesh::materials<dimension> &materials) const {
    // Verify total number of material
    if (materials.n_materials != total_materials.nmaterials) {
      FAIL() << "Total number of materials mismatch. "
             << "Expected: " << total_materials.nmaterials << ", "
             << "Got: " << materials.n_materials << std::endl;
    }

    // Verify total number of elements
    if (materials.nspec != total_materials.nelements) {
      FAIL() << "Total number of elements mismatch. "
             << "Expected: " << total_materials.nelements << ", "
             << "Got: " << materials.nspec << std::endl;
    }

    for (const auto &expected : element_materials) {
      // Check if the element ID is within valid range
      if (expected.element_id < 0 || expected.element_id >= materials.nspec) {
        FAIL() << "Element ID " << expected.element_id << " is out of range."
               << std::endl;
      }

      // Get the material type for the element and verify it matches expected
      const auto [medium_tag, property_tag] =
          materials.get_material_type(expected.element_id);

      if (medium_tag != expected.medium_tag ||
          property_tag != expected.property_tag) {
        FAIL() << "Material type mismatch for element " << expected.element_id
               << ". "
               << "Expected: ("
               << specfem::element::to_string(expected.medium_tag) << ", "
               << specfem::element::to_string(expected.property_tag) << "), "
               << "Got: (" << specfem::element::to_string(medium_tag) << ", "
               << specfem::element::to_string(property_tag) << ")" << std::endl;
      }

      // Retrieve the material and verify it matches expected
      // Note: Update this macro when adding new material types
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC)), {
            if (medium_tag == _medium_tag_ && property_tag == _property_tag_) {
              const auto computed_material =
                  materials.get_material<_medium_tag_, _property_tag_>(
                      expected.element_id);
              const auto expected_material =
                  std::any_cast<specfem::medium::material<
                      _dimension_tag_, _medium_tag_, _property_tag_> >(
                      expected.material);
              if (computed_material != expected_material) {
                FAIL() << "Material mismatch for element "
                       << expected.element_id << ". "
                       << "Expected: " << expected_material.print() << ", "
                       << "Got: " << computed_material.print() << std::endl;
              }
              break;
            }
          });
    }

    SUCCEED() << "All expected materials are present and correct." << std::endl;

    return;
  }
};

} // namespace specfem::test_configuration

using namespace specfem::test_configuration;

static const std::unordered_map<std::string, ExpectedMaterials3D>
    expected_materials_map = { {
        "EightNodeElastic",
        ExpectedMaterials3D(
            TotalMaterials(1, 8),
            { // Element 0
              ElementMaterial(specfem::element::medium_tag::elastic,
                              specfem::element::property_tag::isotropic, 0,
                              specfem::medium::material<
                                  specfem::dimension::type::dim3,
                                  specfem::element::medium_tag::elastic,
                                  specfem::element::property_tag::isotropic>(
                                  2300.0, 1500.0, 2800.0, 2444.4, 300.0, 0.0)),
              // Element 5
              ElementMaterial(
                  specfem::element::medium_tag::elastic,
                  specfem::element::property_tag::isotropic, 5,
                  specfem::medium::material<
                      specfem::dimension::type::dim3,
                      specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic>(
                      2300.0, 1500.0, 2800.0, 2444.4, 300.0, 0.0)) }),
        // Add more test cases as needed
    } };

TEST_P(Mesh3DTest, Materials) {
  const auto &param_name = GetParam();
  if (expected_materials_map.find(param_name) == expected_materials_map.end()) {
    GTEST_SKIP() << "No ground truth defined for test case: " << param_name
                 << std::endl;
    return;
  }

  const auto &mesh = getMesh();
  const auto &materials = mesh.materials;
  const auto &expected = expected_materials_map.at(param_name);
  expected.check(materials);
}
