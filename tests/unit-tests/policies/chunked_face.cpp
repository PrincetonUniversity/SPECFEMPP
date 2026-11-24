#include "../SPECFEM_Environment.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/element.hpp"
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
          specfem::element::dimension_tag::dim3, Kokkos::DefaultExecutionSpace>;

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
  specfem::mesh_entity::element<specfem::element::dimension_tag::dim3> elem;

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
        KOKKOS_CLASS_LAMBDA(const typename decltype(iterator)::base_index_type
                                &iterator_index) {
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
  specfem::mesh_entity::element<specfem::element::dimension_tag::dim3> elem{
    ngll
  };
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
  static constexpr int N = 10;
  static constexpr int n_points = 5;
};

TEST_F(FaceViewTest, Construction) {
  specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace> faces("test_faces",
                                                                   N, n_points);
  EXPECT_EQ(faces.N, N);
  EXPECT_EQ(faces.n_points, n_points);
}

TEST_F(FaceViewTest, SubviewExtraction) {
  using FaceViewType =
      specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>;
  FaceViewType faces("test_faces", N, n_points);

  // Initialize some data
  Kokkos::parallel_for("init",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                       InitFaceViewIndicesFunctor<FaceViewType>(faces));
  Kokkos::fence();

  // Get a subview
  auto subview = faces(Kokkos::make_pair(2, 5));
  EXPECT_EQ(subview.N, 3);
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
  FaceViewType faces("test_faces", N, n_points);

  // Initialize
  Kokkos::parallel_for("init",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
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

namespace chunked_face_test_impl {

using point_index_type =
    specfem::point::face_index<specfem::element::dimension_tag::dim3>;

/// Face type for face f, cycling through the six dim3 types.
KOKKOS_INLINE_FUNCTION
specfem::mesh_entity::dim3::type face_type_for(const int f) {
  switch (f % 6) {
  case 0:
    return specfem::mesh_entity::dim3::type::bottom;
  case 1:
    return specfem::mesh_entity::dim3::type::top;
  case 2:
    return specfem::mesh_entity::dim3::type::front;
  case 3:
    return specfem::mesh_entity::dim3::type::back;
  case 4:
    return specfem::mesh_entity::dim3::type::left;
  default:
    return specfem::mesh_entity::dim3::type::right;
  }
}

/// Maps a flat index over N * n_points * n_points to (f, i, j).
KOKKOS_INLINE_FUNCTION
void decode_point(const int linear, const int n_points, int &f, int &i,
                  int &j) {
  const int points_per_face = n_points * n_points;
  f = linear / points_per_face;
  const int rem = linear % points_per_face;
  i = rem / n_points;
  j = rem % n_points;
}

/// Number of fields of a that differ from b.
KOKKOS_INLINE_FUNCTION
int count_field_mismatches(const point_index_type &a,
                           const point_index_type &b) {
  int mismatches = 0;
  mismatches += (a.ispec != b.ispec);
  mismatches += (a.iface != b.iface);
  mismatches += (a.ipoint_i != b.ipoint_i);
  mismatches += (a.ipoint_j != b.ipoint_j);
  mismatches += (a.iz != b.iz);
  mismatches += (a.iy != b.iy);
  mismatches += (a.ix != b.ix);
  mismatches += (a.face_type != b.face_type);
  return mismatches;
}

/// Number of fields of a that differ from the closed-form fill formula.
KOKKOS_INLINE_FUNCTION
int count_formula_mismatches(const point_index_type &a, const int f,
                             const int i, const int j) {
  int mismatches = 0;
  mismatches += (a.ispec != 100 + f);
  mismatches += (a.iface != f);
  mismatches += (a.ipoint_i != i);
  mismatches += (a.ipoint_j != j);
  mismatches += (a.iz != f + 10 * i + 100 * j);
  mismatches += (a.iy != 1000 + f + 10 * i + 100 * j);
  mismatches += (a.ix != 2000 + f + 10 * i + 100 * j);
  mismatches += (a.face_type != face_type_for(f));
  return mismatches;
}

/// Fills every face and point with a distinct, formula-defined value.
template <typename FaceViewType> struct FillDistinctFaceViewFunctor {
  FaceViewType faces;
  int n_points;

  FillDistinctFaceViewFunctor(FaceViewType faces_, int n_points_)
      : faces(faces_), n_points(n_points_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int f) const {
    faces.element_index(f) = 100 + f;
    faces.face_index(f) = f;
    faces.face_types(f) = face_type_for(f);
    for (int i = 0; i < n_points; ++i) {
      for (int j = 0; j < n_points; ++j) {
        faces.iz(f, i, j) = f + 10 * i + 100 * j;
        faces.iy(f, i, j) = 1000 + f + 10 * i + 100 * j;
        faces.ix(f, i, j) = 2000 + f + 10 * i + 100 * j;
      }
    }
  }
};

/// Counts mismatches between faces(f, i, j) and faces(f)(i, j).
template <typename FaceViewType> struct CompareFaceAccessFunctor {
  FaceViewType faces;
  int n_points;

  CompareFaceAccessFunctor(FaceViewType faces_, int n_points_)
      : faces(faces_), n_points(n_points_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int linear, int &mismatches) const {
    int f, i, j;
    decode_point(linear, n_points, f, i, j);
    const point_index_type a = faces(f, i, j);
    const point_index_type b = faces(f)(i, j);
    mismatches += count_field_mismatches(a, b);
  }
};

/// Counts mismatches between faces(f, i, j) and the fill formula.
template <typename FaceViewType> struct CompareFormulaFunctor {
  FaceViewType faces;
  int n_points;

  CompareFormulaFunctor(FaceViewType faces_, int n_points_)
      : faces(faces_), n_points(n_points_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int linear, int &mismatches) const {
    int f, i, j;
    decode_point(linear, n_points, f, i, j);
    const point_index_type a = faces(f, i, j);
    mismatches += count_formula_mismatches(a, f, i, j);
  }
};

/// Records (range.first, range.second, nfaces) for each chunk, one row per
/// league rank. Written as a functor rather than a lambda because TEST
/// bodies are private member functions, which CUDA extended lambdas reject.
template <typename ChunkIndexType> struct RecordChunkRangeFunctor {
  Kokkos::View<int *[3], Kokkos::DefaultExecutionSpace> out;

  RecordChunkRangeFunctor(
      const Kokkos::View<int *[3], Kokkos::DefaultExecutionSpace> &out_)
      : out(out_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ChunkIndexType &chunk_index) const {
    const auto &team = chunk_index.get_policy_index();
    const auto range = chunk_index.get_range();
    const int row = team.league_rank();
    out(row, 0) = static_cast<int>(range.first);
    out(row, 1) = static_cast<int>(range.second);
    out(row, 2) = chunk_index.nfaces();
  }
};

} // namespace chunked_face_test_impl

/**
 * @brief Point access face_view(f, i, j) must agree with face_view(f)(i, j)
 * and with the closed-form fill values.
 */
TEST_F(FaceViewTest, PointAccessMatchesFaceAccess) {
  using FaceViewType =
      specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>;
  using namespace chunked_face_test_impl;
  FaceViewType faces("test_faces", N, n_points);

  Kokkos::parallel_for(
      "fill_distinct_faces",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      FillDistinctFaceViewFunctor<FaceViewType>(faces, n_points));
  Kokkos::fence();

  const int total = N * n_points * n_points;

  int mismatches_against_face_access = 0;
  Kokkos::parallel_reduce(
      "compare_point_access_vs_face_access",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, total),
      CompareFaceAccessFunctor<FaceViewType>(faces, n_points),
      mismatches_against_face_access);
  Kokkos::fence();

  int mismatches_against_formula = 0;
  Kokkos::parallel_reduce(
      "compare_point_access_vs_formula",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, total),
      CompareFormulaFunctor<FaceViewType>(faces, n_points),
      mismatches_against_formula);
  Kokkos::fence();

  EXPECT_EQ(mismatches_against_face_access, 0)
      << "faces(f, i, j) differs from faces(f)(i, j) in some field";
  EXPECT_EQ(mismatches_against_formula, 0)
      << "faces(f, i, j) differs from the closed-form fill formula";
}

/**
 * @brief Chunk ranges for 10 faces with chunk size 4 must be
 * (0,4), (4,8), (8,10) with nfaces 4, 4, 2.
 */
TEST(ChunkFaceIndexTest, RangeWithTail) {
  using FaceViewType =
      specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>;
  using Config = specfem::parallel_configuration::face_chunk_config<
      specfem::element::dimension_tag::dim3, 4, Kokkos::DefaultExecutionSpace>;

  constexpr int n_faces = 10;
  constexpr int n_points = 5;

  FaceViewType faces("test_chunk_faces", n_faces, n_points);
  Kokkos::parallel_for(
      "init_chunk_faces",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_faces),
      InitFaceViewFullFunctor<FaceViewType>(faces, n_points));
  Kokkos::fence();

  specfem::execution::ChunkedFaceIterator iterator(Config(), faces);
  using IteratorType = decltype(iterator);

  Kokkos::View<int *[3], Kokkos::DefaultExecutionSpace> out("out", 3);
  specfem::execution::for_each_level(
      "test_chunk_face_range", iterator,
      chunked_face_test_impl::RecordChunkRangeFunctor<
          typename IteratorType::index_type>(out));
  Kokkos::fence();

  auto h_out = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h_out, out);

  const int expected[3][3] = { { 0, 4, 4 }, { 4, 8, 4 }, { 8, 10, 2 } };
  for (int chunk = 0; chunk < 3; ++chunk) {
    for (int k = 0; k < 3; ++k) {
      EXPECT_EQ(h_out(chunk, k), expected[chunk][k])
          << "chunk " << chunk << " field " << k
          << " (0 = range.first, 1 = range.second, 2 = nfaces)";
    }
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
