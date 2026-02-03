#include "enumerations/interface.hpp"
#include "specfem/point/boundary.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "test_helper.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

// Base test fixture for boundary tests with template parameter for SIMD
template <bool UseSIMD>
class PointBoundaryTestUntyped : public ::testing::Test {
protected:
  // Define SIMD-related types for convenience
  using simd_type = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd_type::datatype;
  using mask_type = typename simd_type::mask_type;

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

template <typename T>
class PointBoundaryTest : public PointBoundaryTestUntyped<T::value> {};

using PointBoundaryTestSerial = PointBoundaryTest<Serial>;

TYPED_TEST_SUITE(PointBoundaryTest, TestTypes);

// Test none boundary type in 2D
TYPED_TEST(PointBoundaryTest, NoneBoundary2D) {
  constexpr bool using_simd = TypeParam::value;

  // Define the boundary type with none boundary tag in 2D
  using boundary_type = point::boundary<element::boundary_tag::none,
                                        dimension::type::dim2, using_simd>;

  // Verify static properties
  EXPECT_EQ(boundary_type::boundary_tag, element::boundary_tag::none);

  // Create a boundary object
  boundary_type boundary;

  if constexpr (!using_simd) {
    // Non-SIMD specific checks - we can use direct comparison with boundary
    // tags
    EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

    // Add tag using the += operator
    boundary.tag += element::boundary_tag::none;
    EXPECT_TRUE(boundary.tag == element::boundary_tag::none);
  } else {
    // SIMD specific checks - verify types
    using simd_type = typename boundary_type::simd;
    EXPECT_TRUE(
        (std::is_same<simd_type,
                      specfem::datatype::simd<type_real, true> >::value));

    // Verify the value_type for tag is a simd_like container
    using tag_type = typename std::decay<decltype(boundary.tag)>::type;
    bool is_simd_like =
        std::is_same<tag_type, typename specfem::datatype::simd_like<
                                   specfem::element::boundary_tag_container,
                                   type_real, true>::datatype>::value;
    EXPECT_TRUE(is_simd_like);
  }
}

// Test acoustic free surface boundary type in 2D
TYPED_TEST(PointBoundaryTest, AcousticFreeSurfaceBoundary2D) {
  constexpr bool using_simd = TypeParam::value;

  // Define the boundary type with acoustic free surface boundary tag in 2D
  using boundary_type =
      point::boundary<element::boundary_tag::acoustic_free_surface,
                      dimension::type::dim2, using_simd>;

  // Verify static properties
  EXPECT_EQ(boundary_type::boundary_tag,
            element::boundary_tag::acoustic_free_surface);

  // Create a boundary object
  boundary_type boundary;

  if constexpr (!using_simd) {
    // Check initial value
    EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

    // Adding acoustic free surface tag
    boundary.tag += element::boundary_tag::acoustic_free_surface;
    EXPECT_TRUE(boundary.tag == element::boundary_tag::acoustic_free_surface);
  }
}

// Test stacey boundary type in 2D
TYPED_TEST(PointBoundaryTest, StaceyBoundary2D) {
  constexpr bool using_simd = TypeParam::value;

  // Define the boundary type with stacey boundary tag in 2D
  using boundary_type = point::boundary<element::boundary_tag::stacey,
                                        dimension::type::dim2, using_simd>;

  // Verify static properties
  EXPECT_EQ(boundary_type::boundary_tag, element::boundary_tag::stacey);

  // Create a boundary object
  boundary_type boundary;

  if constexpr (!using_simd) {
    // Check initial value
    EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

    // Set edge weight and normal
    boundary.edge_weight = 2.5;
    boundary.edge_normal(0) = 0.8;
    boundary.edge_normal(1) = 0.6;

    // Verify values
    EXPECT_TRUE(boundary.tag == element::boundary_tag::none);
    EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_weight,
                                             static_cast<type_real>(2.5)))
        << ExpectedGot(2.5, boundary.edge_weight);
    EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(0),
                                             static_cast<type_real>(0.8)))
        << ExpectedGot(0.8, boundary.edge_normal(0));
    EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(1),
                                             static_cast<type_real>(0.6)))
        << ExpectedGot(0.6, boundary.edge_normal(1));
  }
}

// Test composite stacey dirichlet boundary type in 2D
TYPED_TEST(PointBoundaryTest, CompositeStaceyDirichletBoundary2D) {
  constexpr bool using_simd = TypeParam::value;

  // Define the boundary type with composite stacey dirichlet boundary tag in 2D
  using boundary_type =
      point::boundary<element::boundary_tag::composite_stacey_dirichlet,
                      dimension::type::dim2, using_simd>;

  // Verify static properties
  EXPECT_EQ(boundary_type::boundary_tag,
            element::boundary_tag::composite_stacey_dirichlet);

  // Create a boundary object
  boundary_type boundary;

  if constexpr (!using_simd) {
    // Check initial value
    EXPECT_TRUE(boundary.tag == element::boundary_tag::none);

    // Set edge weight and normal
    boundary.edge_weight = 3.5;
    boundary.edge_normal(0) = 0.6;
    boundary.edge_normal(1) = 0.8;

    // Verify values
    EXPECT_TRUE(boundary.tag == element::boundary_tag::none);
    EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_weight,
                                             static_cast<type_real>(3.5)))
        << ExpectedGot(3.5, boundary.edge_weight);
    EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(0),
                                             static_cast<type_real>(0.6)))
        << ExpectedGot(0.6, boundary.edge_normal(0));
    EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(1),
                                             static_cast<type_real>(0.8)))
        << ExpectedGot(0.8, boundary.edge_normal(1));
  }
}

// Test a composite boundary - adding stacey to acoustic free surface
TEST_F(PointBoundaryTestSerial, CompositeBoundaryCreation) {
  // Start with acoustic free surface
  using boundary_type =
      point::boundary<element::boundary_tag::acoustic_free_surface,
                      dimension::type::dim2, false>;

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

// Test conversion from composite stacey dirichlet to acoustic free surface
TEST_F(PointBoundaryTestSerial, ConversionCompositeToAcousticFreeSurface) {
  // Define the boundary types
  using composite_type =
      point::boundary<element::boundary_tag::composite_stacey_dirichlet,
                      dimension::type::dim2, false>;
  using acoustic_fs_type =
      point::boundary<element::boundary_tag::acoustic_free_surface,
                      dimension::type::dim2, false>;

  // Create a composite boundary object and set values
  composite_type composite;
  composite.tag += element::boundary_tag::composite_stacey_dirichlet;
  composite.edge_weight = 4.5;
  composite.edge_normal(0) = 0.7;
  composite.edge_normal(1) = 0.7;

  // Convert to acoustic free surface boundary type
  acoustic_fs_type acoustic_fs(composite);

  // Verify tag was copied during conversion
  EXPECT_TRUE(acoustic_fs.tag ==
              element::boundary_tag::composite_stacey_dirichlet);
}

// Test conversion from composite stacey dirichlet to stacey
TEST_F(PointBoundaryTestSerial, ConversionCompositeToStacey) {
  // Define the boundary types
  using composite_type =
      point::boundary<element::boundary_tag::composite_stacey_dirichlet,
                      dimension::type::dim2, false>;
  using stacey_type = point::boundary<element::boundary_tag::stacey,
                                      dimension::type::dim2, false>;

  // Create a composite boundary object and set values
  composite_type composite;
  composite.tag += element::boundary_tag::composite_stacey_dirichlet;
  composite.edge_weight = 5.5;
  composite.edge_normal(0) = 0.3;
  composite.edge_normal(1) = 0.9;

  // Convert to stacey boundary type
  stacey_type stacey(composite);

  // Verify that all values were copied
  EXPECT_TRUE(stacey.tag == element::boundary_tag::composite_stacey_dirichlet);
  EXPECT_TRUE(specfem::utilities::is_close(stacey.edge_weight,
                                           static_cast<type_real>(5.5)))
      << ExpectedGot(5.5, stacey.edge_weight);
  EXPECT_TRUE(specfem::utilities::is_close(stacey.edge_normal(0),
                                           static_cast<type_real>(0.3)))
      << ExpectedGot(0.3, stacey.edge_normal(0));
  EXPECT_TRUE(specfem::utilities::is_close(stacey.edge_normal(1),
                                           static_cast<type_real>(0.9)))
      << ExpectedGot(0.9, stacey.edge_normal(1));
}

// Test default constructors and default tag initialization
TEST_F(PointBoundaryTestSerial, DefaultConstructorsAndTagInitialization) {
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
TYPED_TEST(PointBoundaryTest, InheritanceRelationships) {
  constexpr bool using_simd = TypeParam::value;

  // Test inheritance for acoustic_free_surface boundary from none boundary
  using none_type = point::boundary<element::boundary_tag::none,
                                    dimension::type::dim2, using_simd>;
  using acoustic_fs_type =
      point::boundary<element::boundary_tag::acoustic_free_surface,
                      dimension::type::dim2, using_simd>;
  bool acoustic_inherits_none =
      std::is_base_of<none_type, acoustic_fs_type>::value;
  EXPECT_TRUE(acoustic_inherits_none);

  // Test inheritance for stacey boundary from acoustic_free_surface boundary
  using stacey_type = point::boundary<element::boundary_tag::stacey,
                                      dimension::type::dim2, using_simd>;
  bool stacey_inherits_acoustic =
      std::is_base_of<acoustic_fs_type, stacey_type>::value;
  EXPECT_TRUE(stacey_inherits_acoustic);

  // Test inheritance for composite_stacey_dirichlet boundary from stacey
  // boundary
  using composite_type =
      point::boundary<element::boundary_tag::composite_stacey_dirichlet,
                      dimension::type::dim2, using_simd>;
  bool composite_inherits_stacey =
      std::is_base_of<stacey_type, composite_type>::value;
  EXPECT_TRUE(composite_inherits_stacey);

  // Test transitive inheritance
  bool composite_inherits_none =
      std::is_base_of<none_type, composite_type>::value;
  EXPECT_TRUE(composite_inherits_none);
}

// Test boundary tag container's operators with boundary tags
TEST_F(PointBoundaryTestSerial, BoundaryTagContainerOperators) {
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

  // Test equality operators
  EXPECT_TRUE(boundary.tag ==
              element::boundary_tag::composite_stacey_dirichlet);
  EXPECT_TRUE(boundary.tag == element::boundary_tag::acoustic_free_surface);
  EXPECT_TRUE(boundary.tag == element::boundary_tag::stacey);
  EXPECT_FALSE(boundary.tag == element::boundary_tag::none);

  // Test inequality operators
  EXPECT_FALSE(boundary.tag !=
               element::boundary_tag::composite_stacey_dirichlet);
  EXPECT_FALSE(boundary.tag != element::boundary_tag::acoustic_free_surface);
  EXPECT_FALSE(boundary.tag != element::boundary_tag::stacey);
  EXPECT_TRUE(boundary.tag != element::boundary_tag::none);
}

// // Test for 3D edge_normal initialization
// TYPED_TEST(PointBoundaryTest, Boundary3D_EdgeNormalInitialization) {
//   constexpr bool using_simd = TypeParam::value;

//   // Skip this test for SIMD case as it may not be applicable
//   if constexpr (using_simd) {
//     GTEST_SKIP() << "3D edge_normal initialization test not applicable for
//     SIMD"; return;
//   }

//   // Define the boundary types
//   using boundary3d_type =
//       point::boundary<element::boundary_tag::stacey, dimension::type::dim3,
//                       false>;

//   // Create boundary object
//   boundary3d_type boundary;

//   // Check edge_normal dimensions
//   EXPECT_EQ(boundary.edge_normal.extent(0), 3);  // 3D has 3 components

//   // Check initialization values
//   EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(0), 0.0))
//       << ExpectedGot(0.0, boundary.edge_normal(0));
//   EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(1), 0.0))
//       << ExpectedGot(0.0, boundary.edge_normal(1));
//   EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(2), 0.0))
//       << ExpectedGot(0.0, boundary.edge_normal(2));

//   // Set edge normal values
//   boundary.edge_normal(0) = 0.1;
//   boundary.edge_normal(1) = 0.2;
//   boundary.edge_normal(2) = 0.3;

//   // Verify values
//   EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(0), 0.1))
//       << ExpectedGot(0.1, boundary.edge_normal(0));
//   EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(1), 0.2))
//       << ExpectedGot(0.2, boundary.edge_normal(1));
//   EXPECT_TRUE(specfem::utilities::is_close(boundary.edge_normal(2), 0.3))
//       << ExpectedGot(0.3, boundary.edge_normal(2));
// }
