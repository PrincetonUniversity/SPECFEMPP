/**
 * @file properties.cpp
 * @brief 3D spectral element material properties validation tests
 *
 * This file implements comprehensive testing for the 3D assembly properties
 * system in SPECFEM++. It validates material property assignment, access
 * patterns, and data consistency across different element types and GLL
 * quadrature point configurations.
 *
 * The testing framework provides:
 * - Material property validation at individual GLL points
 * - Support for multiple medium types (elastic, acoustic, poroelastic)
 * - Property type verification (isotropic, anisotropic)
 * - Extensible test case framework using parameterized testing
 * - Integration with assembly structure validation
 *
 * @see specfem::assembly::properties for 3D properties implementation
 * @see specfem::point::properties for individual point property access
 */
#include "../test_fixture.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include <any>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <gtest/gtest.h>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

namespace specfem::assembly_test {

/**
 * @brief 3D Gauss-Lobatto-Legendre (GLL) grid configuration
 *
 * Represents the quadrature grid dimensions for 3D spectral elements.
 * Each dimension defines the number of GLL points used for numerical
 * integration and interpolation in that coordinate direction.
 */
struct GLLGrid {
  int ngllz; ///< Number of GLL points in z-direction
  int nglly; ///< Number of GLL points in y-direction
  int ngllx; ///< Number of GLL points in x-direction

  GLLGrid() = default;

  /**
   * @brief Construct GLL grid with specified dimensions
   *
   * @param ngllz Number of GLL points in z-direction
   * @param nglly Number of GLL points in y-direction
   * @param ngllx Number of GLL points in x-direction
   */
  GLLGrid(int ngllz, int nglly, int ngllx)
      : ngllz(ngllz), nglly(nglly), ngllx(ngllx) {}
};

/**
 * @brief Test specification for 3D material properties at a GLL point
 *
 * Encapsulates expected material property data for a specific GLL quadrature
 * point within a spectral element. Used to define reference values for
 * property validation tests.
 *
 * This structure combines spatial location (element and GLL indices) with
 * material classification (medium and property types) and the actual
 * property values for comparison against computed results.
 */
struct Properties3D {
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;

  specfem::element::medium_tag medium_tag; ///< Medium type (elastic, acoustic,
                                           ///< etc.)
  specfem::element::property_tag property_tag; ///< Property type (isotropic,
                                               ///< anisotropic)

  int ispec; ///< Spectral element index
  int iz;    ///< GLL point index in z-direction
  int iy;    ///< GLL point index in y-direction
  int ix;    ///< GLL point index in x-direction

  std::any property; ///< Type-erased property object for flexible storage

  Properties3D() = default;

  /**
   * @brief Construct property specification for a GLL point
   *
   * @tparam PropertyType Specific property type (e.g., elastic isotropic)
   * @param ispec Spectral element index
   * @param iz GLL point z-index
   * @param iy GLL point y-index
   * @param ix GLL point x-index
   * @param medium_tag Physical medium classification
   * @param property_tag Material property classification
   * @param property Actual property values to validate against
   */
  template <typename PropertyType>
  Properties3D(int ispec, int iz, int iy, int ix,
               specfem::element::medium_tag medium_tag,
               specfem::element::property_tag property_tag,
               const PropertyType &property)
      : ispec(ispec), iz(iz), iy(iy), ix(ix), medium_tag(medium_tag),
        property_tag(property_tag), property(property) {}
};

/**
 * @brief Expected properties validation data for 3D assembly tests
 *
 * Contains complete test specification including mesh dimensions, GLL grid
 * configuration, and expected property values at specific quadrature points.
 * Provides validation methods to verify assembly properties against reference
 * data with comprehensive error reporting.
 *
 * This structure serves as the primary test oracle for material property
 * correctness in the 3D assembly system.
 */
struct ExpectedProperties3D {
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;

  int nspec;                                 ///< Number of spectral elements
  GLLGrid gll_grid;                          ///< GLL quadrature grid dimensions
  std::vector<Properties3D> properties_list; ///< Expected property values

  ExpectedProperties3D() = default;

  /**
   * @brief Construct expected properties specification
   *
   * @param nspec Number of spectral elements in the test mesh
   * @param gll_grid GLL quadrature grid configuration
   * @param properties_list List of expected property values at specific points
   */
  ExpectedProperties3D(
      int nspec, const GLLGrid &gll_grid,
      const std::initializer_list<Properties3D> &properties_list)
      : nspec(nspec), gll_grid(gll_grid), properties_list(properties_list) {}

  /**
   * @brief Validate assembly properties against expected values
   *
   * Performs comprehensive validation of the properties system including:
   * - Mesh dimension verification (nspec, GLL grid)
   * - Individual property value validation at specified points
   * - Type-safe property access and comparison
   * - Detailed error reporting for mismatches
   *
   * @param properties Assembly properties object to validate
   */
  void check(const specfem::assembly::properties<dimension> &properties) const {
    // Validate mesh dimensions
    if (properties.nspec != nspec) {
      FAIL() << "Number of spectral elements mismatch. Expected: " << nspec
             << ", Got: " << properties.nspec << std::endl;
    }

    // Validate GLL grid configuration
    if (properties.element_grid.ngllz != gll_grid.ngllz ||
        properties.element_grid.nglly != gll_grid.nglly ||
        properties.element_grid.ngllx != gll_grid.ngllx) {
      FAIL() << "GLL grid dimensions mismatch. Expected: (" << gll_grid.ngllz
             << ", " << gll_grid.nglly << ", " << gll_grid.ngllx << "), Got: ("
             << properties.element_grid.ngllz << ", "
             << properties.element_grid.nglly << ", "
             << properties.element_grid.ngllx << ")" << std::endl;
    }

    // Validate individual property specifications
    for (const auto &expected : properties_list) {
      // Validate element and GLL point indices
      if (expected.ispec < 0 || expected.ispec >= properties.nspec) {
        FAIL() << "Element ID " << expected.ispec << " is out of range."
               << std::endl;
      }

      if (expected.iz < 0 || expected.iz >= properties.element_grid.ngllz) {
        FAIL() << "GLL index iz " << expected.iz << " is out of range."
               << std::endl;
      }

      if (expected.iy < 0 || expected.iy >= properties.element_grid.nglly) {
        FAIL() << "GLL index iy " << expected.iy << " is out of range."
               << std::endl;
      }

      if (expected.ix < 0 || expected.ix >= properties.element_grid.ngllx) {
        FAIL() << "GLL index ix " << expected.ix << " is out of range."
               << std::endl;
      }

      // Type-safe property validation using template metaprogramming
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim3) * MEDIUM_SET(elastic) * PROPERTY_SET(isotropic),
          [&]<typename ElementTags>() {
            if (ElementTags::medium_tag == expected.medium_tag &&
                ElementTags::property_tag == expected.property_tag) {
              const int ispec = expected.ispec;
              const int iz = expected.iz;
              const int iy = expected.iy;
              const int ix = expected.ix;

              // Create point index for property access
              const auto index =
                  specfem::point::index<ElementTags::dimension_tag, false>(
                      ispec, iz, iy, ix);

              // Load computed property from assembly structure
              const auto computed_property = [&]() {
                specfem::point::properties<specfem::tags::Tags<
                    ElementTags::dimension_tag, ElementTags::medium_tag,
                    ElementTags::property_tag, false>>
                    prop_accessor;
                specfem::assembly::load_on_host(index, properties,
                                                prop_accessor);
                return prop_accessor;
              }();

              // Extract expected property from type-erased storage
              const auto expected_property =
                  std::any_cast<specfem::point::properties<specfem::tags::Tags<
                      ElementTags::dimension_tag, ElementTags::medium_tag,
                      ElementTags::property_tag, false>>>(expected.property);

              // Compare properties with detailed error reporting
              if (computed_property != expected_property) {
                FAIL() << "Property mismatch for element " << ispec
                       << " at GLL point (" << iz << ", " << iy << ", " << ix
                       << "). "
                       << "Expected: " << expected_property.print()
                       << ", Got: " << computed_property.print() << std::endl;
              }
            }
          });
    }

    SUCCEED() << "All expected properties are present and correct."
              << std::endl;
  }
};

} // namespace specfem::assembly_test

using namespace specfem::assembly_test;

/**
 * @brief Test case database for 3D assembly property validation
 *
 * Maps test case names to their expected property specifications. Each entry
 * defines a complete test scenario including mesh configuration, GLL grid
 * setup, and reference property values at specific quadrature points.
 *
 * **Current Test Cases:**
 * - **EightNodeElastic**: 8-element elastic isotropic mesh with 5×5×5 GLL grid
 *   - Tests basic elastic material property assignment
 *   - Validates property access at corner GLL points
 *   - Reference values: λ=11.132 GPa, μ=5.175 GPa, ρ=2300 kg/m³
 *
 * @note Additional test cases can be added by extending this map with new
 *       ExpectedProperties3D specifications.
 */
std::unordered_map<std::string, ExpectedProperties3D>
    expected_properties_map = {
      { "EightNodeElastic",
        ExpectedProperties3D(
            8, GLLGrid(5, 5, 5),
            {
                // Element 0: Corner GLL point with elastic isotropic properties
                Properties3D(
                    0, 0, 0, 0, specfem::element::medium_tag::elastic,
                    specfem::element::property_tag::isotropic,
                    specfem::point::properties<specfem::tags::Tags<
                        specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic, false>>(
                        11.132e9, 5.175e9, 2300)) // κ, μ, ρ
                                                  // Add more GLL points as
                                                  // needed
            }) }
    };

/**
 * @brief Parameterized test for 3D assembly material properties validation
 *
 * This test validates the material properties system in the 3D assembly
 * structure by comparing computed property values against reference data
 * at specific GLL quadrature points.
 *
 * **Test Process:**
 * 1. Retrieves assembly structure from test fixture for the given parameter
 * 2. Extracts expected properties specification from test case database
 * 3. Validates mesh dimensions and GLL grid configuration
 * 4. Performs point-wise property comparison using type-safe accessors
 * 5. Reports detailed information on any mismatches found
 *
 * **Validation Coverage:**
 * - Material property values (elastic moduli, density, etc.)
 * - Property access patterns through assembly interface
 * - Type safety of property accessor systems
 * - Mesh dimension and GLL grid consistency
 *
 * **Mathematical Context:**
 * For elastic isotropic materials, validates:
 * - Bulk modulus κ and shear modulus μ (elastic moduli in Pa)
 * - Density ρ (in kg/m³)
 * - Proper assignment to GLL quadrature points
 *
 * @note Test cases are defined in expected_properties_map and must match
 *       the parameterized test fixture parameter names.
 */
TEST_P(Assembly3DTest, Properties) {
  const auto &param_name = GetParam();

  // Check if test case data is available
  if (expected_properties_map.find(param_name) ==
      expected_properties_map.end()) {
    GTEST_SKIP() << "No expected properties data available for test case '"
                 << param_name << "'.";
    return;
  }

  // Get assembly structure and extract properties
  const auto &assembly = getAssembly();
  const auto &properties = assembly.properties;
  const auto &expected_properties = expected_properties_map.at(param_name);

  // Perform comprehensive validation
  expected_properties.check(properties);
}

/**
 * @brief Round-trip write/read test for the 3D property writer/reader.
 *
 * Mirrors the 2D `properties_io_routines` test: fills every property host view
 * with a known value, writes the model to disk with the ASCII backend, corrupts
 * the host views, reads the model back, and verifies the recovered values.
 */
TEST_P(Assembly3DTest, PropertiesIORoutines) {
  // Mutable access to the fixture's assembly (the public accessor is const).
  auto &assembly = this->assembly.assembly;

  const std::string temp_io_directory =
      boost::filesystem::current_path().string() +
      "/tests/unit-tests/temp_properties_io_dim3";
  boost::filesystem::create_directories(temp_io_directory);

  const auto for_each_property_view = [&](auto fn) {
    specfem::tag_dispatch::for_each(
        DIMENSION_SET(dim3) * MEDIUM_SET(elastic, acoustic) *
            PROPERTY_SET(isotropic),
        [&]<typename ElementTags>() {
          constexpr auto medium_tag = ElementTags::medium_tag;
          constexpr auto property_tag = ElementTags::property_tag;
          const auto elements = assembly.element_types.get_elements_on_host(
              medium_tag, property_tag);
          if (elements.size() == 0)
            return;
          const auto container =
              assembly.properties.get_container<medium_tag, property_tag>();
          container.for_each_host_view(
              [&](const auto view, const std::string) { fn(view); });
        });
  };

  try {
    const type_real random_value = 10.1;

    // 1. Set every property host view to a known value and push to device.
    for_each_property_view(
        [&](const auto view) { Kokkos::deep_copy(view, random_value); });
    assembly.properties.copy_to_device();

    // 2. Write the model to disk.
    specfem::io::property_writer<
        specfem::io_backends::ASCII<specfem::io::write>>
        writer(temp_io_directory);
    writer.write(assembly);

    // 3. Corrupt the host views so a successful read is unambiguous.
    for_each_property_view([&](const auto view) {
      Kokkos::deep_copy(view, static_cast<type_real>(-999.0));
    });
    assembly.properties.copy_to_device();

    // 4. Read the model back.
    specfem::io::property_reader<specfem::io_backends::ASCII<specfem::io::read>>
        reader(temp_io_directory);
    reader.read(assembly);

    // 5. Verify the recovered host values. Iterate logical extents (not the
    // flat base span, which may include tiling padding the reader does not
    // restore).
    for_each_property_view([&](const auto view) {
      for (std::size_t e = 0; e < view.extent(0); ++e)
        for (std::size_t iz = 0; iz < view.extent(1); ++iz)
          for (std::size_t iy = 0; iy < view.extent(2); ++iy)
            for (std::size_t ix = 0; ix < view.extent(3); ++ix) {
              EXPECT_NEAR(view(e, iz, iy, ix), random_value, 1e-4);
            }
    });
  } catch (std::exception &e) {
    boost::filesystem::remove_all(temp_io_directory);
    FAIL() << "3D property IO round-trip failed: " << e.what();
  }

  boost::filesystem::remove_all(temp_io_directory);
}
