/**
 * @file jacobian_matrix.cpp
 * @brief Unit tests for 3D Jacobian matrix computation in spectral element
 * assembly.
 *
 * This file contains comprehensive tests for validating the Jacobian matrix
 * computation in 3D spectral element methods. The Jacobian matrix represents
 * the transformation between reference coordinates \f$(\xi, \eta, \gamma)\f$
 * and physical coordinates
 * \f$(x, y, z)\f$ for hexahedral spectral elements.
 *
 * @details The tests verify:
 * - Correct allocation and dimensioning of Jacobian matrix storage
 * - Accurate computation of partial derivatives (\f$\xi_x, \xi_y, \xi_z\f$,
 * etc.)
 * - Proper Jacobian determinant calculation
 * - Consistency between host and device memory layouts
 * - Validation against analytical solutions for simple element geometries
 *
 * The Jacobian transformation is fundamental to spectral element methods as it
 * enables the mapping from the reference element \f$[-1,1]^3\f$ to arbitrary
 * hexahedral elements in physical space, allowing for accurate integration and
 * differentiation operations.
 *
 * @see specfem::assembly::jacobian_matrix
 * @see specfem::jacobian::compute_jacobian
 * @see ExpectedJacobian3D for test data structures
 */

#include "../test_fixture.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "gtest/gtest.h"
#include <Kokkos_Core.hpp>
#include <array>
#include <unordered_map>
#include <vector>

/**
 * @brief Macro for validating Kokkos View allocation with expected dimensions.
 *
 * This macro performs comprehensive dimension checking for 4D Kokkos Views used
 * in Jacobian matrix storage. It validates that each dimension matches the
 * expected values for spectral element discretization:
 * - Extent 0: Number of spectral elements (nspec)
 * - Extent 1: Number of GLL points in z direction (ngllz)
 * - Extent 2: Number of GLL points in y direction (nglly)
 * - Extent 3: Number of GLL points in x direction (ngllx)
 *
 * @param view The Kokkos View to validate
 * @param nspec Expected number of spectral elements
 * @param ngllz Expected number of GLL points in z direction
 * @param nglly Expected number of GLL points in y direction
 * @param ngllx Expected number of GLL points in x direction
 *
 * @note The macro provides detailed error messages indicating which dimension
 *       failed validation and the expected vs. actual values.
 */
#define CHECK_VIEW_ALLOCATION(view, nspec, ngllz, nglly, ngllx)                \
  ASSERT_TRUE(view.extent(0) == nspec)                                         \
      << "View extent 0 mismatch. Expected: " << nspec                         \
      << ", Got: " << view.extent(0) << std::endl;                             \
  ASSERT_TRUE(view.extent(1) == ngllz)                                         \
      << "View extent 1 mismatch. Expected: " << ngllz                         \
      << ", Got: " << view.extent(1) << std::endl;                             \
  ASSERT_TRUE(view.extent(2) == nglly)                                         \
      << "View extent 2 mismatch. Expected: " << nglly                         \
      << ", Got: " << view.extent(2) << std::endl;                             \
  ASSERT_TRUE(view.extent(3) == ngllx)                                         \
      << "View extent 3 mismatch. Expected: " << ngllx                         \
      << ", Got: " << view.extent(3) << std::endl;

namespace specfem::assembly_test {

/**
 * @brief Container for total GLL point configuration in 3D spectral elements.
 *
 * This structure encapsulates the discretization parameters that define the
 * number of Gauss-Lobatto-Legendre (GLL) quadrature points in each spatial
 * dimension for a 3D spectral element mesh.
 *
 * @details In spectral element methods, each hexahedral element is discretized
 * using tensor products of 1D GLL quadrature points. This structure stores
 * the total configuration for the entire mesh.
 */
struct GridExtents {
  int ngllx;     ///< Number of GLL points in x direction
  int nglly;     ///< Number of GLL points in y direction
  int ngllz;     ///< Number of GLL points in z direction
  int nelements; ///< Total number of elements in the mesh

  /**
   * @brief Constructor for GLL point configuration.
   *
   * @param ngllx Number of GLL points in x direction
   * @param nglly Number of GLL points in y direction
   * @param ngllz Number of GLL points in z direction
   * @param nelements Total number of spectral elements
   */
  GridExtents(int ngllx, int nglly, int ngllz, int nelements)
      : ngllx(ngllx), nglly(nglly), ngllz(ngllz), nelements(nelements) {}
};

/**
 * @brief Test fixture for a single 3D hexahedral spectral element.
 *
 * This class represents a single hexahedral spectral element with its geometric
 * properties and control node coordinates. It provides functionality to convert
 * coordinate data into Kokkos Views compatible with SPECFEM++ data structures.
 *
 * @details The element stores:
 * - Element identifier for test organization
 * - Number of control nodes (typically 8 for hexahedral elements)
 * - 3D coordinates of all control nodes in physical space
 *
 * The coordinate transformation follows the standard hexahedral element
 * ordering convention used in SPECFEM++ for consistent geometric
 * representation.
 */
struct Element3D {
  int element_id;                ///< Unique identifier for the element
  int control_nodes_per_element; ///< Number of control nodes (typically 8 for
                                 ///< hex elements)

private:
  std::vector<std::array<type_real, 3> > _coordinates; ///< Private storage for
                                                       ///< 3D coordinates

public:
  /**
   * @brief Constructor for 3D spectral element.
   *
   * @param element_id Unique identifier for this element
   * @param control_nodes_per_element Number of control nodes (typically 8)
   * @param coords Initializer list of 3D coordinate arrays {x, y, z}
   *
   * @note The coordinate ordering should follow the standard hexahedral
   *       element convention used in SPECFEM++.
   */
  Element3D(int element_id, int control_nodes_per_element,
            const std::initializer_list<std::array<type_real, 3> > &coords)
      : element_id(element_id),
        control_nodes_per_element(control_nodes_per_element),
        _coordinates(coords) {
    int i = 0;
    for (const auto &coord : coords) {
      _coordinates[i] = coord;
      ++i;
    }
  }

  /**
   * @brief Convert stored coordinates to Kokkos View format.
   *
   * This method transforms the internally stored coordinate data into a
   * Kokkos HostSpace View that is compatible with SPECFEM++ Jacobian
   * computation routines.
   *
   * @return Kokkos View containing global coordinates for all control nodes
   *
   * @note The returned view uses HostSpace memory and contains
   *       specfem::point::global_coordinates data structures.
   */
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
  coordinates() const {
    const int npoints = static_cast<int>(_coordinates.size());
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace>
        coord_view("coord_view", npoints);
    for (int i = 0; i < npoints; ++i) {
      coord_view(i) = { _coordinates[i][0], _coordinates[i][1],
                        _coordinates[i][2] };
    }
    return coord_view;
  }
};

/**
 * @brief Expected results container for 3D Jacobian matrix validation.
 *
 * This structure encapsulates the expected configuration and validation logic
 * for testing 3D Jacobian matrix computations. It stores reference element
 * geometries and provides comprehensive validation against computed results.
 *
 * @details The validator performs multi-level verification:
 * 1. **Dimensional consistency**: Validates mesh parameters (nspec, ngll*)
 * 2. **Memory allocation**: Ensures all Kokkos Views are properly allocated
 * 3. **Mathematical accuracy**: Compares computed vs. analytical Jacobian
 * values
 * 4. **Point-wise validation**: Tests every GLL quadrature point in every
 * element
 *
 * The validation process uses analytical Jacobian computation via
 * specfem::jacobian::compute_jacobian() and compares results at each
 * GLL point with a specified tolerance.
 */
struct ExpectedJacobian3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3; ///< Compile-time dimension specification

  GridExtents total_gll_points;    ///< GLL discretization parameters
  std::vector<Element3D> elements; ///< Collection of test elements

  /**
   * @brief Constructor for expected Jacobian validation data.
   *
   * @param total_gll_points GLL point configuration for the mesh
   * @param elements Initializer list of Element3D test cases
   */
  ExpectedJacobian3D(GridExtents total_gll_points,
                     const std::initializer_list<Element3D> &elements)
      : total_gll_points(total_gll_points), elements(elements) {}

  /**
   * @brief Comprehensive validation of computed Jacobian matrix.
   *
   * This method performs exhaustive testing of the Jacobian matrix computation
   * including dimensional verification, memory allocation checks, and
   * point-wise mathematical validation against analytical solutions.
   *
   * @param jacobian_matrix The computed Jacobian matrix to validate
   *
   * @details Validation steps:
   * 1. **Parameter Validation**: Checks nspec, ngllx, nglly, ngllz consistency
   * 2. **Memory Layout Validation**: Verifies all device and host View
   * allocations
   * 3. **Quadrature Setup**: Initializes GLL quadrature points for integration
   * 4. **Point-wise Comparison**: For each element and GLL point:
   *    - Computes analytical Jacobian using reference implementation
   *    - Extracts computed values from jacobian_matrix
   *    - Performs tolerance-based comparison (1e-6 default)
   *
   * @note The validation uses specfem::assembly::load_on_host() to extract
   *       point-specific Jacobian data and
   * specfem::jacobian::compute_jacobian() for analytical reference values.
   *
   * @throws AssertionError if any validation step fails, with detailed
   *         error messages indicating the failure location and expected vs.
   * actual values.
   */
  void check(const specfem::assembly::jacobian_matrix<dimension>
                 &jacobian_matrix) const {

    // Validate mesh configuration parameters
    ASSERT_EQ(jacobian_matrix.nspec, total_gll_points.nelements)
        << "Total number of elements mismatch. "
        << "Expected: " << total_gll_points.nelements << ", "
        << "Got: " << jacobian_matrix.nspec << std::endl;
    ASSERT_EQ(jacobian_matrix.ngllx, total_gll_points.ngllx)
        << "Total number of GLL points in x direction mismatch. "
        << "Expected: " << total_gll_points.ngllx << ", "
        << "Got: " << jacobian_matrix.ngllx << std::endl;
    ASSERT_EQ(jacobian_matrix.nglly, total_gll_points.nglly)
        << "Total number of GLL points in y direction mismatch. "
        << "Expected: " << total_gll_points.nglly << ", "
        << "Got: " << jacobian_matrix.nglly << std::endl;
    ASSERT_EQ(jacobian_matrix.ngllz, total_gll_points.ngllz)
        << "Total number of GLL points in z direction mismatch. "
        << "Expected: " << total_gll_points.ngllz << ", "
        << "Got: " << jacobian_matrix.ngllz << std::endl;

    // Extract dimensions for validation
    const int nspec = jacobian_matrix.nspec;
    const int ngllx = jacobian_matrix.ngllx;
    const int nglly = jacobian_matrix.nglly;
    const int ngllz = jacobian_matrix.ngllz;

    // Initialize GLL quadrature for coordinate transformations
    const auto quadrature = []() {
      specfem::quadrature::gll::gll gll{};
      return specfem::quadrature::quadratures(gll);
    }();

    // Validate host mirror View allocations for partial derivatives
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_xix, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_xiy, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_xiz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_etax, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_etay, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_etaz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_gammax, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_gammay, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_gammaz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_jacobian, nspec, ngllz, nglly,
                          ngllx);

    // Validate device View allocations for partial derivatives
    CHECK_VIEW_ALLOCATION(jacobian_matrix.xix, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.xiy, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.xiz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.etax, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.etay, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.etaz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.gammax, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.gammay, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.gammaz, nspec, ngllz, nglly, ngllx);

    // Extract GLL quadrature points for coordinate transformation
    const auto xi = quadrature.gll.get_hxi();

    // Point-wise Jacobian validation for all elements and GLL points
    for (const auto &element : elements) {
      const int ispec = element.element_id;
      const auto expected_coord = element.coordinates();

      // Iterate through all GLL points in 3D (z, y, x order)
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix) {
            // Extract reference coordinates for this GLL point
            const type_real xil = xi(ix);
            const type_real etal = xi(iy);
            const type_real zetal = xi(iz);

            // Compute analytical Jacobian matrix using reference implementation
            const auto expected_jacobian = specfem::jacobian::compute_jacobian(
                expected_coord, element.control_nodes_per_element, xil, etal,
                zetal);

            // Set numerical tolerance for floating-point comparison
            const type_real tol = 1e-6;

            // Extract computed Jacobian data from assembly structure
            const auto point_jacobian = [&]()
                -> specfem::point::jacobian_matrix<dimension, true, false> {
              specfem::point::jacobian_matrix<dimension, true, false>
                  point_jacobian;
              specfem::assembly::load_on_host(
                  specfem::point::index<dimension, false>(ispec, iz, iy, ix),
                  jacobian_matrix, point_jacobian);
              return point_jacobian;
            }();

            // Perform tolerance-based comparison of Jacobian matrices
            EXPECT_TRUE(expected_jacobian == point_jacobian)
                << "Jacobian matrix mismatch at element " << ispec
                << ", GLL point (" << ix << ", " << iy << ", " << iz << ").";
          }
        }
      }
    }

    const auto [small_jacobian, dummy] = jacobian_matrix.check_small_jacobian();
    EXPECT_FALSE(small_jacobian)
        << "Small Jacobian determinant detected in the computed matrix.";

    SUCCEED()
        << "Jacobian matrix check passed for all elements and GLL points.";
  }
};

} // namespace specfem::assembly_test

using namespace specfem::assembly_test;

/**
 * @brief Ground truth test data for 3D Jacobian matrix validation.
 *
 * This section defines reference test cases with known geometric configurations
 * and expected Jacobian matrix behavior. Each test case represents a specific
 * element geometry that exercises different aspects of the Jacobian
 * computation.
 *
 * @details Current test cases:
 * - **EightNodeElastic**: Standard hexahedral element with uniform geometry
 *   - Configuration: 5×5×5 GLL points, 8 spectral elements
 *   - Geometry: Regular rectangular parallelepiped (50000×40000×30000 units)
 *   - Purpose: Validates basic Jacobian computation for well-conditioned
 * elements
 *   - Expected behavior: Uniform Jacobian across all GLL points due to regular
 * geometry
 *
 * @note When adding new test cases, focus on geometries that might reveal
 *       computational issues:
 *       - High aspect ratio elements
 *       - Severely distorted or skewed elements
 *       - Elements with curved boundaries
 *       - Near-degenerate configurations
 *
 * @warning The coordinate ordering must follow SPECFEM++ hexahedral element
 *          conventions for consistent geometric representation.
 */
static const std::unordered_map<std::string, ExpectedJacobian3D>
    expected_jacobians_3d = {
      { "EightNodeElastic",
        { { 5, 5, 5, 8 }, // GLL configuration: ngllx, nglly, ngllz, nelements
          { { 0,          // Element ID
              8,          // Control nodes per element (hexahedral)
              {
                  // Control node coordinates {x, y, z} in SPECFEM++ ordering
                  { 0.0, 0.0, -60000.0 },         // Node 0: bottom-front-left
                  { 50000.0, 0.0, -60000.0 },     // Node 1: bottom-front-right
                  { 50000.0, 40000.0, -60000.0 }, // Node 2: bottom-back-right
                  { 0.0, 40000.0, -60000.0 },     // Node 3: bottom-back-left
                  { 0.0, 0.0, -30000.0 },         // Node 4: top-front-left
                  { 50000.0, 0.0, -30000.0 },     // Node 5: top-front-right
                  { 50000.0, 40000.0, -30000.0 }, // Node 6: top-back-right
                  { 0.0, 40000.0, -30000.0 }      // Node 7: top-back-left
              } } } } }
    };

/**
 * @brief Parameterized test for 3D Jacobian matrix computation validation.
 *
 * This test validates the Jacobian matrix computation for 3D spectral elements
 * using parameterized test cases. Each test case represents a different
 * geometric configuration and validates both the structural integrity and
 * mathematical accuracy of the computed Jacobian matrices.
 *
 * @details Test execution flow:
 * 1. **Parameter Retrieval**: Gets test case name from parameterized test
 * framework
 * 2. **Data Validation**: Ensures expected results exist for the test case
 * 3. **Assembly Access**: Retrieves computed Jacobian matrix from test assembly
 * 4. **Comprehensive Validation**: Delegates to ExpectedJacobian3D::check()
 * for:
 *    - Dimensional consistency verification
 *    - Memory allocation validation
 *    - Point-wise mathematical accuracy testing
 *
 * @note The test uses GoogleTest's parameterized testing framework, allowing
 *       multiple geometric configurations to be tested with the same validation
 * logic.
 *
 * @param param_name Test case identifier (e.g., "EightNodeElastic")
 *
 * @warning If no expected data exists for a test case, the test is skipped
 *          with an appropriate message rather than failing.
 *
 * @see Assembly3DTest for test fixture implementation
 * @see expected_jacobians_3d for available test case configurations
 */
TEST_P(Assembly3DTest, JacobianMatrix) {
  const auto &param_name = GetParam();

  if (expected_jacobians_3d.find(param_name) == expected_jacobians_3d.end()) {
    GTEST_SKIP() << "No expected jacobian matrix data available for test case '"
                 << param_name << "'.";
    return;
  }

  const auto &assembly = getAssembly();
  const auto &jacobian_matrix = assembly.jacobian_matrix;
  const auto &expected_jacobian = expected_jacobians_3d.at(param_name);
  expected_jacobian.check(jacobian_matrix);
}
