#include "../SPECFEM_Environment.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly/face_types.hpp"
#include "specfem/execution.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

namespace {

// Functors to replace lambdas for CUDA compatibility
// (CUDA does not allow extended __host__ __device__ lambdas inside
// functions with private/protected access, such as TEST_F's TestBody)

template <typename FaceViewType> struct InitFaceViewIndicesFunctor {
  FaceViewType faces;

  InitFaceViewIndicesFunctor(FaceViewType faces_) : faces(faces_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    faces.element_index(i) = i;
    faces.face_index(i) = i * 10;
  }
};

template <typename FaceViewType> struct InitFaceViewFullFunctor {
  FaceViewType faces;
  int n_points;

  InitFaceViewFullFunctor(FaceViewType faces_, int n_points_)
      : faces(faces_), n_points(n_points_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    faces.element_index(i) = i;
    faces.face_index(i) = i;
    faces.face_types(i) = specfem::mesh_entity::dim3::type::bottom;
    for (int j = 0; j < n_points; ++j) {
      for (int k = 0; k < n_points; ++k) {
        faces.iz(i, j, k) = 0;
        faces.iy(i, j, k) = j;
        faces.ix(i, j, k) = k;
      }
    }
  }
};

template <typename FaceViewType> struct TestFaceAccessFunctor {
  FaceViewType faces;

  TestFaceAccessFunctor(FaceViewType faces_) : faces(faces_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int, int &local_result) const {
    auto face = faces(0);
    auto point = face(2, 3);
    local_result = point.ix + point.iy * 10 + point.iz * 100;
  }
};

} // namespace

// Base fixture for common functionality
class ChunkedFaceIteratorTestBase {
public:
  using ParallelConfig =
      specfem::parallel_configuration::default_chunk_face_config<
          specfem::dimension::type::dim3, Kokkos::DefaultExecutionSpace>;

  constexpr static int num_points = 5; // ngll per face dimension
  // Storage view indexed by [face][ipoint_i][ipoint_j]
  using StorageViewType = Kokkos::View<int ***, Kokkos::DefaultExecutionSpace>;
  using FacesViewType =
      specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>;
};

// Test parameter structs
struct FaceIteratorTestParams {
  std::size_t number_of_faces;
  std::string name;

  FaceIteratorTestParams(std::size_t n, const char *test_name)
      : number_of_faces(n), name(test_name) {}
};

std::ostream &operator<<(std::ostream &os,
                         const FaceIteratorTestParams &params) {
  os << params.name;
  return os;
}

// Fixture for Face Iterator tests
class FaceIterator : public ChunkedFaceIteratorTestBase {
public:
  StorageViewType view;
  FacesViewType faces;
  std::string name;
  int number_of_faces;
  specfem::mesh_entity::element<specfem::dimension::type::dim3> elem;

  FaceIterator(const FaceIteratorTestParams &params)
      : view("view", params.number_of_faces, num_points, num_points),
        faces("faces", params.number_of_faces, num_points), name(params.name),
        number_of_faces(params.number_of_faces), elem(num_points) {

    this->reset();
    Kokkos::fence();
  }

  void run() const {
    specfem::execution::ChunkedFaceIterator iterator(ParallelConfig(),
                                                     this->faces);
    specfem::execution::for_all(
        "test_chunked_face_iterator", iterator,
        KOKKOS_CLASS_LAMBDA(const typename decltype(
            iterator)::base_index_type &iterator_index) {
          const auto index = iterator_index.get_index();
          Kokkos::atomic_add(&view(index.iface, index.ipoint_i, index.ipoint_j),
                             1);
        });

    Kokkos::fence();
  }

  void check() const {
    auto host_view = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(host_view, view);

    for (int i = 0; i < number_of_faces; ++i) {
      for (int j = 0; j < num_points; ++j) {
        for (int k = 0; k < num_points; ++k) {
          EXPECT_EQ(host_view(i, j, k), 1)
              << "Face iterator failed at face " << i << " point (" << j << ","
              << k << ") "
              << "for test: " << name;
        }
      }
    }
  }

  void reset() const {
    // Initialize storage view to zeros
    Kokkos::parallel_for(
        "initialize_storage",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_faces),
        KOKKOS_CLASS_LAMBDA(const int i) {
          for (int j = 0; j < num_points; ++j) {
            for (int k = 0; k < num_points; ++k) {
              view(i, j, k) = 0;
            }
          }
        });

    // Initialize faces view - cycle through face types
    constexpr auto bottom = specfem::mesh_entity::dim3::type::bottom;
    constexpr auto top = specfem::mesh_entity::dim3::type::top;
    constexpr auto front = specfem::mesh_entity::dim3::type::front;
    constexpr auto back = specfem::mesh_entity::dim3::type::back;
    constexpr auto left = specfem::mesh_entity::dim3::type::left;
    constexpr auto right = specfem::mesh_entity::dim3::type::right;

    Kokkos::parallel_for(
        "initialize_faces",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_faces),
        KOKKOS_CLASS_LAMBDA(const int i) {
          faces.element_index(i) = static_cast<int>(i);
          faces.face_index(i) = static_cast<int>(i);
          // Cycle through face types
          switch (i % 6) {
          case 0:
            faces.face_types(i) = bottom;
            break;
          case 1:
            faces.face_types(i) = top;
            break;
          case 2:
            faces.face_types(i) = front;
            break;
          case 3:
            faces.face_types(i) = back;
            break;
          case 4:
            faces.face_types(i) = left;
            break;
          case 5:
            faces.face_types(i) = right;
            break;
          }

          // Set up quadrature point coordinates based on face type
          for (int ipoint_i = 0; ipoint_i < num_points; ++ipoint_i) {
            for (int ipoint_j = 0; ipoint_j < num_points; ++ipoint_j) {
              int iz_val, iy_val, ix_val;
              elem.get_face_coordinates(faces.face_types(i), ipoint_i, ipoint_j,
                                        iz_val, iy_val, ix_val);
              faces.iz(i, ipoint_i, ipoint_j) = iz_val;
              faces.iy(i, ipoint_i, ipoint_j) = iy_val;
              faces.ix(i, ipoint_i, ipoint_j) = ix_val;
            }
          }
        });
  }
};

// Value parameterized tests
class FaceIteratorTest
    : public ::testing::TestWithParam<FaceIteratorTestParams> {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(FaceIteratorTest, VisitAllPoints) {
  const FaceIterator test(GetParam());
  test.reset();
  Kokkos::fence();
  test.run();
  test.check();
}

TEST_P(FaceIteratorTest, MultipleIterations) {
  // Test that running the iterator multiple times visits each point multiple
  // times
  const FaceIterator test(GetParam());
  test.reset();
  Kokkos::fence();
  test.run();
  test.run();
  test.run();

  auto host_view = Kokkos::create_mirror_view(test.view);
  Kokkos::deep_copy(host_view, test.view);

  for (int i = 0; i < test.number_of_faces; ++i) {
    for (int j = 0; j < test.num_points; ++j) {
      for (int k = 0; k < test.num_points; ++k) {
        EXPECT_EQ(host_view(i, j, k), 3)
            << "Face iterator multiple iterations failed at face " << i
            << " point (" << j << "," << k << ") "
            << "for test: " << test.name;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    FaceIteratorTests, FaceIteratorTest,
    ::testing::Values(FaceIteratorTestParams{ 10, "SmallFaceValues" },
                      FaceIteratorTestParams{ 100, "MediumFaceValues" },
                      FaceIteratorTestParams{ 1000, "LargeFaceValues" },
                      FaceIteratorTestParams{ 16,
                                              "ExactChunkSizeFaceValues" }));

/**
 * @brief Test face index coordinate mapping
 *
 * Verifies that face points are correctly mapped to element coordinates
 * for all six face types
 */
class FaceCoordinateMappingTest : public ::testing::Test {
protected:
  static constexpr int ngll = 5;
  specfem::mesh_entity::element<specfem::dimension::type::dim3> elem{ ngll };
};

TEST_F(FaceCoordinateMappingTest, BottomFace) {
  using namespace specfem::mesh_entity::dim3;
  for (int i = 0; i < ngll; ++i) {
    for (int j = 0; j < ngll; ++j) {
      int iz, iy, ix;
      elem.get_face_coordinates(type::bottom, i, j, iz, iy, ix);
      EXPECT_EQ(iz, 0) << "Bottom face z-coordinate should be 0";
      EXPECT_EQ(iy, i) << "Bottom face y-coordinate should be ipoint_i";
      EXPECT_EQ(ix, j) << "Bottom face x-coordinate should be ipoint_j";
    }
  }
}

TEST_F(FaceCoordinateMappingTest, TopFace) {
  using namespace specfem::mesh_entity::dim3;
  for (int i = 0; i < ngll; ++i) {
    for (int j = 0; j < ngll; ++j) {
      int iz, iy, ix;
      elem.get_face_coordinates(type::top, i, j, iz, iy, ix);
      EXPECT_EQ(iz, ngll - 1) << "Top face z-coordinate should be ngll-1";
      EXPECT_EQ(iy, i) << "Top face y-coordinate should be ipoint_i";
      EXPECT_EQ(ix, j) << "Top face x-coordinate should be ipoint_j";
    }
  }
}

TEST_F(FaceCoordinateMappingTest, FrontFace) {
  using namespace specfem::mesh_entity::dim3;
  for (int i = 0; i < ngll; ++i) {
    for (int j = 0; j < ngll; ++j) {
      int iz, iy, ix;
      elem.get_face_coordinates(type::front, i, j, iz, iy, ix);
      EXPECT_EQ(iz, i) << "Front face z-coordinate should be ipoint_i";
      EXPECT_EQ(iy, 0) << "Front face y-coordinate should be 0";
      EXPECT_EQ(ix, j) << "Front face x-coordinate should be ipoint_j";
    }
  }
}

TEST_F(FaceCoordinateMappingTest, BackFace) {
  using namespace specfem::mesh_entity::dim3;
  for (int i = 0; i < ngll; ++i) {
    for (int j = 0; j < ngll; ++j) {
      int iz, iy, ix;
      elem.get_face_coordinates(type::back, i, j, iz, iy, ix);
      EXPECT_EQ(iz, i) << "Back face z-coordinate should be ipoint_i";
      EXPECT_EQ(iy, ngll - 1) << "Back face y-coordinate should be ngll-1";
      EXPECT_EQ(ix, j) << "Back face x-coordinate should be ipoint_j";
    }
  }
}

TEST_F(FaceCoordinateMappingTest, LeftFace) {
  using namespace specfem::mesh_entity::dim3;
  for (int i = 0; i < ngll; ++i) {
    for (int j = 0; j < ngll; ++j) {
      int iz, iy, ix;
      elem.get_face_coordinates(type::left, i, j, iz, iy, ix);
      EXPECT_EQ(iz, i) << "Left face z-coordinate should be ipoint_i";
      EXPECT_EQ(iy, j) << "Left face y-coordinate should be ipoint_j";
      EXPECT_EQ(ix, 0) << "Left face x-coordinate should be 0";
    }
  }
}

TEST_F(FaceCoordinateMappingTest, RightFace) {
  using namespace specfem::mesh_entity::dim3;
  for (int i = 0; i < ngll; ++i) {
    for (int j = 0; j < ngll; ++j) {
      int iz, iy, ix;
      elem.get_face_coordinates(type::right, i, j, iz, iy, ix);
      EXPECT_EQ(iz, i) << "Right face z-coordinate should be ipoint_i";
      EXPECT_EQ(iy, j) << "Right face y-coordinate should be ipoint_j";
      EXPECT_EQ(ix, ngll - 1) << "Right face x-coordinate should be ngll-1";
    }
  }
}

/**
 * @brief Test FaceView structure
 */
class FaceViewTest : public ::testing::Test {
protected:
  static constexpr int n_faces = 10;
  static constexpr int n_points = 5;
};

TEST_F(FaceViewTest, Construction) {
  specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace> faces(
      "test_faces", n_faces, n_points);
  EXPECT_EQ(faces.n_faces, n_faces);
  EXPECT_EQ(faces.n_points, n_points);
}

TEST_F(FaceViewTest, SubviewExtraction) {
  using FaceViewType =
      specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>;
  FaceViewType faces("test_faces", n_faces, n_points);

  // Initialize some data
  Kokkos::parallel_for(
      "init", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_faces),
      InitFaceViewIndicesFunctor<FaceViewType>(faces));
  Kokkos::fence();

  // Get a subview
  auto subview = faces(Kokkos::make_pair(2, 5));
  EXPECT_EQ(subview.n_faces, 3);
  EXPECT_EQ(subview.n_points, n_points);

  // Verify subview data
  auto h_element_index = Kokkos::create_mirror_view(subview.element_index);
  Kokkos::deep_copy(h_element_index, subview.element_index);

  EXPECT_EQ(h_element_index(0), 2);
  EXPECT_EQ(h_element_index(1), 3);
  EXPECT_EQ(h_element_index(2), 4);
}

TEST_F(FaceViewTest, FaceAccess) {
  using FaceViewType =
      specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>;
  FaceViewType faces("test_faces", n_faces, n_points);

  // Initialize
  Kokkos::parallel_for(
      "init", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_faces),
      InitFaceViewFullFunctor<FaceViewType>(faces, n_points));
  Kokkos::fence();

  // Test accessing a face and its points on device
  int result = 0;
  Kokkos::parallel_reduce(
      "test_face_access",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
      TestFaceAccessFunctor<FaceViewType>(faces), result);
  Kokkos::fence();

  // point should be (iz=0, iy=2, ix=3) for bottom face at (2,3)
  EXPECT_EQ(result, 3 + 2 * 10 + 0 * 100);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
