/**
 * @file tags.cpp
 * @brief 3D mesh element tagging system validation tests
 *
 * This file implements comprehensive testing for the 3D mesh tags system in
 * SPECFEM++. It validates element classification through medium, property, and
 * boundary tag assignment across different mesh configurations and element
 * types.
 *
 * The testing framework provides:
 * - Element tag validation for individual spectral elements
 * - Support for all medium types (elastic, acoustic, poroelastic)
 * - Property type verification (isotropic, anisotropic)
 * - Boundary condition tag validation (none, stacey, free_surface)
 * - Extensible test case framework using parameterized testing
 * - Integration with MESHFEM3D mesh structure validation
 *
 * @see specfem::mesh::meshfem3d::tags for 3D tags implementation
 * @see specfem::mesh::impl::tags_container for individual element tags
 */
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"
#include "specfem_setup.hpp"
#include "test_fixture.hpp"

namespace specfem::test_configuration {

/**
 * @brief Expected element tag specification for validation testing
 *
 * Represents the expected classification tags for a specific spectral element.
 * Used to define reference values for tag validation tests by combining
 * element identification with its expected medium, property, and boundary
 * classification.
 *
 * This structure serves as the test oracle for verifying correct tag
 * assignment in the mesh tagging system.
 */
struct ElementTags {
  int element_id;                          ///< ID of the spectral element
  specfem::element::medium_tag medium_tag; ///< Medium type (elastic, acoustic,
                                           ///< etc.)
  specfem::element::property_tag property_tag; ///< Property type (isotropic,
                                               ///< anisotropic)
  specfem::element::boundary_tag boundary_tag; ///< Boundary condition (none,
                                               ///< stacey, etc.)

  /**
   * @brief Construct element tag specification
   *
   * @param element_id Spectral element identifier
   * @param medium_tag Expected physical medium classification
   * @param property_tag Expected material property classification
   * @param boundary_tag Expected boundary condition classification
   */
  ElementTags(int element_id, specfem::element::medium_tag medium_tag,
              specfem::element::property_tag property_tag,
              specfem::element::boundary_tag boundary_tag)
      : element_id(element_id), medium_tag(medium_tag),
        property_tag(property_tag), boundary_tag(boundary_tag) {}
};

/**
 * @brief Expected tags validation data for 3D mesh tests
 *
 * Contains complete test specification including mesh dimensions and expected
 * tag values for specific elements. Provides validation methods to verify
 * mesh tags against reference data with comprehensive error reporting.
 *
 * This structure serves as the primary test oracle for element tag correctness
 * in the 3D mesh tagging system, enabling validation of material classification
 * and boundary condition assignment.
 */
struct ExpectedTags3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  int nspec;                             ///< Total number of spectral elements
  std::vector<ElementTags> element_tags; ///< List of expected element tags

  /**
   * @brief Construct expected tags specification
   *
   * @param nspec Number of spectral elements in the test mesh
   * @param elements List of expected tag values for specific elements
   */
  ExpectedTags3D(int nspec, const std::initializer_list<ElementTags> &elements)
      : nspec(nspec), element_tags(elements) {}

  /**
   * @brief Validate mesh tags against expected values
   *
   * Performs comprehensive validation of the mesh tagging system including:
   * - Mesh dimension verification (element count)
   * - Individual element tag validation (medium, property, boundary)
   * - Range checking for element identifiers
   * - Detailed error reporting for tag mismatches
   *
   * @param tags Mesh tags object to validate
   */
  void check(const specfem::mesh::tags<dimension> &tags) const {
    // Validate mesh dimensions
    if (tags.nspec != nspec) {
      FAIL() << "Number of spectral elements mismatch. Expected: " << nspec
             << ", Got: " << tags.nspec << std::endl;
    }

    // Validate individual element tags
    for (const auto &expected : element_tags) {
      // Validate element ID range
      if (expected.element_id < 0 || expected.element_id >= tags.nspec) {
        FAIL() << "Element ID " << expected.element_id << " is out of range."
               << std::endl;
      }

      // Extract computed tags from mesh structure
      const auto [medium_tag, property_tag, boundary_tag] =
          tags.tags_container(expected.element_id);

      // Compare all tag components with detailed error reporting
      if (medium_tag != expected.medium_tag ||
          property_tag != expected.property_tag ||
          boundary_tag != expected.boundary_tag) {
        FAIL() << "Tag mismatch for element " << expected.element_id << ". "
               << "Expected: ("
               << specfem::element::to_string(expected.medium_tag) << ", "
               << specfem::element::to_string(expected.property_tag) << ", "
               << specfem::element::to_string(expected.boundary_tag) << "), "
               << "Got: (" << specfem::element::to_string(medium_tag) << ", "
               << specfem::element::to_string(property_tag) << ", "
               << specfem::element::to_string(boundary_tag) << ")" << std::endl;
      }
    }

    SUCCEED() << "All expected tags are present and correct." << std::endl;
  }
};

} // namespace specfem::test_configuration

using namespace specfem::test_configuration;

/**
 * @brief Test case database for 3D mesh tag validation
 *
 * Maps test case names to their expected tag specifications. Each entry defines
 * a complete test scenario including mesh configuration and reference tag
 * values for specific spectral elements that should be validated.
 *
 * **Current Test Cases:**
 * - **EightNodeElastic**: 8-element elastic isotropic mesh validation
 *   - Tests basic elastic material tag assignment
 *   - Validates isotropic property classification
 *   - Verifies boundary_tag::none for interior elements
 *   - Spot-checks elements 0, 1, and 5 for tag correctness
 *
 * **Tag Classification:**
 * - **Medium tags**: Determine wave equation type (elastic → stress-strain
 * relations)
 * - **Property tags**: Define constitutive relations (isotropic → simplified
 * parameters)
 * - **Boundary tags**: Specify boundary treatment (none → no special
 * conditions)
 *
 * @note Additional test cases can be added by extending this map with new
 *       ExpectedTags3D specifications for different mesh configurations.
 */
std::unordered_map<std::string, ExpectedTags3D> expected_tags_map = {
  { "EightNodeElastic",
    ExpectedTags3D(8, { ElementTags(0, specfem::element::medium_tag::elastic,
                                    specfem::element::property_tag::isotropic,
                                    specfem::element::boundary_tag::none),
                        ElementTags(1, specfem::element::medium_tag::elastic,
                                    specfem::element::property_tag::isotropic,
                                    specfem::element::boundary_tag::none),
                        ElementTags(5, specfem::element::medium_tag::elastic,
                                    specfem::element::property_tag::isotropic,
                                    specfem::element::boundary_tag::none) }) }
  // Add more test cases as needed
};

/**
 * @brief Parameterized test for 3D mesh element tag validation
 *
 * This test validates the element tagging system in 3D mesh structures by
 * comparing computed tag assignments against reference data for specific
 * spectral elements.
 *
 * **Test Process:**
 * 1. Retrieves mesh structure from test fixture for the given parameter
 * 2. Extracts expected tag specification from test case database
 * 3. Validates mesh dimensions (element count)
 * 4. Performs element-wise tag comparison for specified elements
 * 5. Reports detailed information on any tag mismatches found
 *
 * **Validation Coverage:**
 * - Medium tag correctness (determines wave equation selection)
 * - Property tag accuracy (defines constitutive relation complexity)
 * - Boundary tag assignment (specifies boundary condition treatment)
 * - Element identifier range validation
 * - Mesh dimension consistency
 *
 * **Physical Significance:**
 * Tag validation ensures proper element classification for:
 * - Wave equation solver selection based on medium type
 * - Material property matrix construction based on property type
 * - Boundary condition application based on boundary type
 * - Assembly structure initialization and computational kernel dispatch
 *
 * @note Test cases are defined in expected_tags_map and must match the
 *       parameterized test fixture parameter names. Missing test data
 *       results in test skip with appropriate messaging.
 */
TEST_P(Mesh3DTest, Tags) {
  const auto &param_name = GetParam();

  // Check if test case data is available
  if (expected_tags_map.find(param_name) == expected_tags_map.end()) {
    GTEST_SKIP() << "No ground truth defined for test case: " << param_name
                 << std::endl;
    return;
  }

  // Get mesh structure and extract tags
  const auto &mesh = getMesh();
  const auto &tags = mesh.tags;
  const auto &expected = expected_tags_map.at(param_name);

  // Perform comprehensive tag validation
  expected.check(tags);
}
