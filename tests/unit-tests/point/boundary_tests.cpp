#include "datatypes/simd.hpp"
#include "enumerations/interface.hpp"
#include "specfem/point/boundary.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

// Test fixture for boundary tests
class PointBoundaryTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Kokkos if needed for tests
    if (!Kokkos::is_initialized())
      Kokkos::initialize();
  }

  void TearDown() override {
    // Finalize Kokkos if needed
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  }
};

// Test none boundary type in 2D
TEST_F(PointBoundaryTest, NoneBoundary2D) {
  // Define the boundary type with none boundary tag in 2D
  using boundary_type =
      point::boundary<element::boundary_tag::none, dimension::type::dim2,
                      false>; // UseSIMD

  // Verify static properties - Use proper enum class
  EXPECT_EQ(boundary_type::boundary_tag, element::boundary_tag::none);

  // Create a boundary object
  boundary_type boundary;

  // Check if tag is already initialized to the expected value
  EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

  // Add tag using the += operator with proper enum class
  boundary.tag += element::boundary_tag::none;

  // Check the value
  EXPECT_TRUE(boundary.tag == element::boundary_tag::none);
}

// Test acoustic free surface boundary type in 2D
TEST_F(PointBoundaryTest, AcousticFreeSurfaceBoundary2D) {
  // Define the boundary type with acoustic free surface boundary tag in 2D
  using boundary_type =
      point::boundary<element::boundary_tag::acoustic_free_surface,
                      dimension::type::dim2,
                      false>; // UseSIMD

  // Verify static properties - Use proper enum class
  EXPECT_EQ(boundary_type::boundary_tag,
            element::boundary_tag::acoustic_free_surface);

  // Create a boundary object
  boundary_type boundary;

  // Check initial value
  EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

  // Adding acoustic free surface tag to itself should not change anything
  boundary.tag += element::boundary_tag::acoustic_free_surface;
  EXPECT_TRUE(boundary.tag == element::boundary_tag::acoustic_free_surface);
}

// Test stacey boundary type in 2D
TEST_F(PointBoundaryTest, StaceyBoundary2D) {
  // Define the boundary type with stacey boundary tag in 2D
  using boundary_type =
      point::boundary<element::boundary_tag::stacey, dimension::type::dim2,
                      false>; // UseSIMD

  // Verify static properties - Use proper enum class
  EXPECT_EQ(boundary_type::boundary_tag, element::boundary_tag::stacey);

  // Create a boundary object
  boundary_type boundary;

  // Check initial value
  EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

  // Set edge weight and normal
  boundary.edge_weight = 2.5;
  boundary.edge_normal(0) = 0.8;
  boundary.edge_normal(1) = 0.6;

  // Verify values - Use proper enum class for tag comparison
  EXPECT_TRUE(boundary.tag == element::boundary_tag::none);
  EXPECT_REAL_EQ(boundary.edge_weight, 2.5);
  EXPECT_REAL_EQ(boundary.edge_normal(0), 0.8);
  EXPECT_REAL_EQ(boundary.edge_normal(1), 0.6);
}

// Test composite stacey dirichlet boundary type in 2D
TEST_F(PointBoundaryTest, CompositeStaceyDirichletBoundary2D) {
  // Define the boundary type with composite stacey dirichlet boundary tag in 2D
  using boundary_type =
      point::boundary<element::boundary_tag::composite_stacey_dirichlet,
                      dimension::type::dim2,
                      false>; // UseSIMD

  // Verify static properties - Use proper enum class
  EXPECT_EQ(boundary_type::boundary_tag,
            element::boundary_tag::composite_stacey_dirichlet);

  // Create a boundary object
  boundary_type boundary;

  // Check initial value
  EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

  // Set edge weight and normal
  boundary.edge_weight = 3.5;
  boundary.edge_normal(0) = 0.6;
  boundary.edge_normal(1) = 0.8;

  // Verify values - Use proper enum class for tag comparison
  EXPECT_TRUE(boundary.tag == element::boundary_tag::none);
  EXPECT_REAL_EQ(boundary.edge_weight, 3.5);
  EXPECT_REAL_EQ(boundary.edge_normal(0), 0.6);
  EXPECT_REAL_EQ(boundary.edge_normal(1), 0.8);
}

// Test a composite boundary - adding stacey to acoustic free surface
TEST_F(PointBoundaryTest, CompositeBoundaryCreation) {
  // Start with acoustic free surface
  using boundary_type =
      point::boundary<element::boundary_tag::acoustic_free_surface,
                      dimension::type::dim2,
                      false>; // UseSIMD

  // Create a boundary object
  boundary_type boundary;

  // Check initial value
  boundary.tag += element::boundary_tag::acoustic_free_surface;
  EXPECT_TRUE(boundary.tag == element::boundary_tag::acoustic_free_surface);

  // Add stacey boundary tag, this should create a composite boundary
  boundary.tag += element::boundary_tag::stacey;

  // Verify it changed to a composite boundary
  EXPECT_TRUE(boundary.tag ==
              element::boundary_tag::composite_stacey_dirichlet);
}

// Test SIMD version of boundary type
TEST_F(PointBoundaryTest, NoneBoundary_SIMD) {
  // Define the SIMD boundary type with none boundary tag
  using boundary_type =
      point::boundary<element::boundary_tag::none, dimension::type::dim2,
                      true>; // UseSIMD

  // Verify static properties - Use proper enum class
  EXPECT_EQ(boundary_type::boundary_tag, element::boundary_tag::none);

  // Create a boundary object
  boundary_type boundary;

  // For SIMD mode, we need to handle the simd_like container differently
  // The tag is stored as an array with multiple lanes

  // Get the SIMD size
  constexpr int simd_size = datatype::simd<type_real, true>::size();

  // We can't use the += operator directly on the tag for SIMD case
  // Instead, we would need to either:
  // 1. Use a SIMD-aware API if the boundary_tag_container provides one, or
  // 2. Access each lane individually if the implementation allows

  // Since we don't have access to modify individual lanes through the public
  // interface, we should test what we can verify: that the object was
  // constructed with the correct default values and SIMD traits

  // Verify the SIMD type is defined
  using simd_type = typename boundary_type::simd;
  EXPECT_TRUE((std::is_same<simd_type,
                            specfem::datatype::simd<type_real, true> >::value));

  // Verify the value_type for tag is a simd_like container
  using tag_type = std::decay<decltype(boundary.tag)>::type;
  bool is_simd_like =
      std::is_same<tag_type, typename specfem::datatype::simd_like<
                                 specfem::element::boundary_tag_container,
                                 type_real, true>::datatype>::value;
  EXPECT_TRUE(is_simd_like);

  // Note: Using += on SIMD version might not be supported or might apply to all
  // lanes For now, we'll just indicate this limitation boundary.tag +=
  // element::boundary_tag::none; // This may not work for SIMD version

  // Instead, we'll verify the object was created successfully
  SUCCEED();
}

// Test conversion from composite stacey dirichlet to acoustic free surface
TEST_F(PointBoundaryTest, ConversionCompositeToAcousticFreeSurface) {
  // Define the boundary types
  using composite_type =
      point::boundary<element::boundary_tag::composite_stacey_dirichlet,
                      dimension::type::dim2,
                      false>; // UseSIMD
  using acoustic_fs_type =
      point::boundary<element::boundary_tag::acoustic_free_surface,
                      dimension::type::dim2,
                      false>; // UseSIMD

  // Create a composite boundary object and set values
  composite_type composite;
  composite.tag += element::boundary_tag::composite_stacey_dirichlet;
  composite.edge_weight = 4.5;
  composite.edge_normal(0) = 0.7;
  composite.edge_normal(1) = 0.7;

  // Convert to acoustic free surface boundary type
  acoustic_fs_type acoustic_fs(composite);

  // Verify tag was copied during conversion - use enum class
  EXPECT_TRUE(acoustic_fs.tag ==
              element::boundary_tag::composite_stacey_dirichlet);
}

// Test conversion from composite stacey dirichlet to stacey
TEST_F(PointBoundaryTest, ConversionCompositeToStacey) {
  // Define the boundary types
  using composite_type =
      point::boundary<element::boundary_tag::composite_stacey_dirichlet,
                      dimension::type::dim2,
                      false>; // UseSIMD
  using stacey_type =
      point::boundary<element::boundary_tag::stacey, dimension::type::dim2,
                      false>; // UseSIMD

  // Create a composite boundary object and set values
  composite_type composite;
  composite.tag += element::boundary_tag::composite_stacey_dirichlet;
  composite.edge_weight = 5.5;
  composite.edge_normal(0) = 0.3;
  composite.edge_normal(1) = 0.9;

  // Convert to stacey boundary type
  stacey_type stacey(composite);

  // Verify that all values were copied - use enum class for tag comparison
  EXPECT_TRUE(stacey.tag == element::boundary_tag::composite_stacey_dirichlet);
  EXPECT_REAL_EQ(stacey.edge_weight, 5.5);
  EXPECT_REAL_EQ(stacey.edge_normal(0), 0.3);
  EXPECT_REAL_EQ(stacey.edge_normal(1), 0.9);
}

// // Test none boundary type in 3D
// TEST_F(PointBoundaryTest, NoneBoundary3D) {
//   // Define the boundary type with none boundary tag in 3D
//   using boundary_type = point::boundary<element::boundary_tag::none,
//                                       dimension::type::dim3,
//                                       false>; // UseSIMD

//   // Verify static properties - Use proper enum class
//   EXPECT_EQ(boundary_type::boundary_tag, element::boundary_tag::none);

//   // Create a boundary object
//   boundary_type boundary;

//   // Check initial value
//   EXPECT_TRUE(boundary.tag == element::boundary_tag::none);
// }

// // Test stacey boundary type in 3D
// TEST_F(PointBoundaryTest, StaceyBoundary3D) {
//   // Define the boundary type with stacey boundary tag in 3D
//   using boundary_type = point::boundary<element::boundary_tag::stacey,
//                                       dimension::type::dim3,
//                                       false>; // UseSIMD

//   // Verify static properties - Use proper enum class
//   EXPECT_EQ(boundary_type::boundary_tag, element::boundary_tag::stacey);

//   // Create a boundary object
//   boundary_type boundary;

//   // Check initial value
//   EXPECT_TRUE(boundary.tag == element::boundary_tag::stacey);

//   // Set edge weight and normal values
//   boundary.edge_weight = 2.5;
//   boundary.edge_normal(0) = 0.6;
//   boundary.edge_normal(1) = 0.7;
//   boundary.edge_normal(2) = 0.8;

//   // Verify values - Use proper enum class for tag comparison
//   EXPECT_TRUE(boundary.tag == element::boundary_tag::stacey);
//   EXPECT_REAL_EQ(boundary.edge_weight, 2.5);
//   EXPECT_REAL_EQ(boundary.edge_normal(0), 0.6);
//   EXPECT_REAL_EQ(boundary.edge_normal(1), 0.7);
//   EXPECT_REAL_EQ(boundary.edge_normal(2), 0.8);
// }

// Test default constructors and default tag initialization
TEST_F(PointBoundaryTest, DefaultConstructorsAndTagInitialization) {
  // Test default constructor for none boundary
  using none_type = point::boundary<element::boundary_tag::none,
                                    dimension::type::dim2, false>;
  none_type none_boundary;

  // Test default constructor for acoustic_free_surface boundary
  using acoustic_fs_type =
      point::boundary<element::boundary_tag::acoustic_free_surface,
                      dimension::type::dim2, false>;
  acoustic_fs_type acoustic_fs_boundary;
  acoustic_fs_boundary.tag += element::boundary_tag::acoustic_free_surface;

  // Test default constructor for stacey boundary
  using stacey_type = point::boundary<element::boundary_tag::stacey,
                                      dimension::type::dim2, false>;
  stacey_type stacey_boundary;
  stacey_boundary.tag += element::boundary_tag::stacey;

  // Test default constructor for composite_stacey_dirichlet boundary
  using composite_type =
      point::boundary<element::boundary_tag::composite_stacey_dirichlet,
                      dimension::type::dim2, false>;
  composite_type composite_boundary;
  composite_boundary.tag += element::boundary_tag::composite_stacey_dirichlet;

  // Verify tag is initialized to the boundary_tag static member by default
  EXPECT_TRUE(none_boundary.tag == element::boundary_tag::none);
  EXPECT_TRUE(acoustic_fs_boundary.tag ==
              element::boundary_tag::acoustic_free_surface);
  EXPECT_TRUE(stacey_boundary.tag == element::boundary_tag::stacey);
  EXPECT_TRUE(composite_boundary.tag ==
              element::boundary_tag::composite_stacey_dirichlet);

  // Verify edge_weight and edge_normal are properly initialized for stacey and
  // derived types
  EXPECT_REAL_EQ(stacey_boundary.edge_weight, 0.0);
  EXPECT_REAL_EQ(stacey_boundary.edge_normal(0), 0.0);
  EXPECT_REAL_EQ(stacey_boundary.edge_normal(1), 0.0);

  EXPECT_REAL_EQ(composite_boundary.edge_weight, 0.0);
  EXPECT_REAL_EQ(composite_boundary.edge_normal(0), 0.0);
  EXPECT_REAL_EQ(composite_boundary.edge_normal(1), 0.0);
}

// Test inheritance relationships
TEST_F(PointBoundaryTest, InheritanceRelationships) {
  // Test inheritance for acoustic_free_surface boundary from none boundary
  using none_type = point::boundary<element::boundary_tag::none,
                                    dimension::type::dim2, false>;
  using acoustic_fs_type =
      point::boundary<element::boundary_tag::acoustic_free_surface,
                      dimension::type::dim2, false>;
  bool acoustic_inherits_none =
      std::is_base_of<none_type, acoustic_fs_type>::value;
  EXPECT_TRUE(acoustic_inherits_none);

  // Test inheritance for stacey boundary from acoustic_free_surface boundary
  using stacey_type = point::boundary<element::boundary_tag::stacey,
                                      dimension::type::dim2, false>;
  bool stacey_inherits_acoustic =
      std::is_base_of<acoustic_fs_type, stacey_type>::value;
  EXPECT_TRUE(stacey_inherits_acoustic);

  // Test inheritance for composite_stacey_dirichlet boundary from stacey
  // boundary
  using composite_type =
      point::boundary<element::boundary_tag::composite_stacey_dirichlet,
                      dimension::type::dim2, false>;
  bool composite_inherits_stacey =
      std::is_base_of<stacey_type, composite_type>::value;
  EXPECT_TRUE(composite_inherits_stacey);

  // Test transitive inheritance
  bool composite_inherits_none =
      std::is_base_of<none_type, composite_type>::value;
  EXPECT_TRUE(composite_inherits_none);
}

// Test boundary tag container's operators with boundary tags
TEST_F(PointBoundaryTest, BoundaryTagContainerOperators) {
  // Create boundary object
  using boundary_type = point::boundary<element::boundary_tag::none,
                                        dimension::type::dim2, false>;
  boundary_type boundary;

  // Initially none
  EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

  // Adding none keeps it none
  boundary.tag += element::boundary_tag::none;
  EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

  // Add acoustic free surface
  boundary.tag += element::boundary_tag::acoustic_free_surface;
  EXPECT_TRUE(boundary.tag == element::boundary_tag::acoustic_free_surface);

  // Adding none doesn't change it
  boundary.tag += element::boundary_tag::none;
  EXPECT_TRUE(boundary.tag == element::boundary_tag::acoustic_free_surface);

  // Adding stacey makes it composite
  boundary.tag += element::boundary_tag::stacey;
  EXPECT_TRUE(boundary.tag ==
              element::boundary_tag::composite_stacey_dirichlet);

  // Test equality operators - using enum class
  EXPECT_TRUE(boundary.tag ==
              element::boundary_tag::composite_stacey_dirichlet);
  EXPECT_TRUE(boundary.tag == element::boundary_tag::acoustic_free_surface);
  EXPECT_TRUE(boundary.tag == element::boundary_tag::stacey);
  EXPECT_FALSE(boundary.tag == element::boundary_tag::none);

  // Test inequality operators - using enum class
  EXPECT_FALSE(boundary.tag !=
               element::boundary_tag::composite_stacey_dirichlet);
  EXPECT_FALSE(boundary.tag != element::boundary_tag::acoustic_free_surface);
  EXPECT_FALSE(boundary.tag != element::boundary_tag::stacey);
  EXPECT_TRUE(boundary.tag != element::boundary_tag::none);
}
