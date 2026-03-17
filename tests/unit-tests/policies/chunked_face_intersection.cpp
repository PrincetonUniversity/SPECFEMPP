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

// Base fixture for common functionality
class ChunkedFaceIntersectionTestBase {
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
struct FaceIntersectionTestParams {
  std::size_t number_of_faces;
  std::string name;

  FaceIntersectionTestParams(std::size_t n, const char *test_name)
      : number_of_faces(n), name(test_name) {}
};

std::ostream &operator<<(std::ostream &os,
                         const FaceIntersectionTestParams &params) {
  os << params.name;
  return os;
}

// Fixture for Face Intersection Iterator tests
class FaceIntersectionIterator : public ChunkedFaceIntersectionTestBase {
public:
  StorageViewType self_view;
  StorageViewType coupled_view;
  FacesViewType faces;
  FacesViewType intersection_faces;
  std::string name;
  int number_of_faces;
  specfem::mesh_entity::element<specfem::element::dimension_tag::dim3> elem;

  FaceIntersectionIterator(const FaceIntersectionTestParams &params)
      : self_view("self_view", params.number_of_faces, num_points, num_points),
        coupled_view("coupled_view", params.number_of_faces, num_points,
                     num_points),
        faces("faces", params.number_of_faces, num_points),
        intersection_faces("intersection_faces", params.number_of_faces,
                           num_points),
        name(params.name), number_of_faces(params.number_of_faces),
        elem(num_points) {

    this->reset();
    Kokkos::fence();
  }

  void run() const {
    specfem::execution::ChunkedIntersectionIterator iterator(
        ParallelConfig(), faces, intersection_faces);
    specfem::execution::for_all(
        "test_chunked_intersection_face_iterator", iterator,
        KOKKOS_CLASS_LAMBDA(const typename decltype(
            iterator)::base_index_type &iterator_index) {
          const auto index = iterator_index.get_index();
          const auto self_index = index.self_index;
          const auto coupled_index = index.coupled_index;
          Kokkos::atomic_add(&self_view(self_index.iface, self_index.ipoint_i,
                                        self_index.ipoint_j),
                             1);
          Kokkos::atomic_add(&coupled_view(coupled_index.iface,
                                           coupled_index.ipoint_i,
                                           coupled_index.ipoint_j),
                             1);
        });
    Kokkos::fence();
  }

  void check() const {
    auto host_self_view = Kokkos::create_mirror_view(self_view);
    Kokkos::deep_copy(host_self_view, self_view);
    auto host_coupled_view = Kokkos::create_mirror_view(coupled_view);
    Kokkos::deep_copy(host_coupled_view, coupled_view);

    for (int i = 0; i < number_of_faces; ++i) {
      for (int j = 0; j < num_points; ++j) {
        for (int k = 0; k < num_points; ++k) {
          EXPECT_EQ(host_self_view(i, j, k), 1)
              << "Self intersection iterator failed at face " << i << " point ("
              << j << "," << k << ") "
              << "expected: 1 for test: " << name;

          EXPECT_EQ(host_coupled_view(i, j, k), 1)
              << "Coupled intersection iterator failed at face " << i
              << " point (" << j << "," << k << ") "
              << "expected: 1 for test: " << name;
        }
      }
    }
  }

  void reset() const {
    // Initialize storage views to zeros
    Kokkos::parallel_for(
        "initialize_storage",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_faces),
        KOKKOS_CLASS_LAMBDA(const int i) {
          for (int j = 0; j < num_points; ++j) {
            for (int k = 0; k < num_points; ++k) {
              self_view(i, j, k) = 0;
              coupled_view(i, j, k) = 0;
            }
          }
        });

    // Initialize faces views
    constexpr auto top = specfem::mesh_entity::dim3::type::top;
    constexpr auto bottom = specfem::mesh_entity::dim3::type::bottom;

    Kokkos::parallel_for(
        "initialize_intersection_faces",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_faces),
        KOKKOS_CLASS_LAMBDA(const int i) {
          // Self faces
          faces.element_index(i) = static_cast<int>(i);
          faces.face_index(i) = static_cast<int>(i);
          faces.face_types(i) = top;

          // Coupled faces (reversed order)
          intersection_faces.element_index(i) =
              static_cast<int>(number_of_faces - i - 1);
          intersection_faces.face_index(i) = static_cast<int>(i);
          intersection_faces.face_types(i) = bottom;

          // Set up quadrature point coordinates
          for (int ipoint_i = 0; ipoint_i < num_points; ++ipoint_i) {
            for (int ipoint_j = 0; ipoint_j < num_points; ++ipoint_j) {
              int iz_val, iy_val, ix_val;

              elem.get_face_coordinates(top, ipoint_i, ipoint_j, iz_val, iy_val,
                                        ix_val);
              faces.iz(i, ipoint_i, ipoint_j) = iz_val;
              faces.iy(i, ipoint_i, ipoint_j) = iy_val;
              faces.ix(i, ipoint_i, ipoint_j) = ix_val;

              elem.get_face_coordinates(bottom, ipoint_i, ipoint_j, iz_val,
                                        iy_val, ix_val);
              intersection_faces.iz(i, ipoint_i, ipoint_j) = iz_val;
              intersection_faces.iy(i, ipoint_i, ipoint_j) = iy_val;
              intersection_faces.ix(i, ipoint_i, ipoint_j) = ix_val;
            }
          }
        });
  }
};

// Value parameterized tests
class FaceIntersectionIteratorTest
    : public ::testing::TestWithParam<FaceIntersectionTestParams> {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(FaceIntersectionIteratorTest, VisitAllPoints) {
  const FaceIntersectionIterator test(GetParam());
  test.reset();
  Kokkos::fence();
  test.run();
  test.check();
}

TEST_P(FaceIntersectionIteratorTest, MultipleIterations) {
  const FaceIntersectionIterator test(GetParam());
  test.reset();
  Kokkos::fence();
  test.run();
  test.run();
  test.run();

  auto host_self_view = Kokkos::create_mirror_view(test.self_view);
  Kokkos::deep_copy(host_self_view, test.self_view);
  auto host_coupled_view = Kokkos::create_mirror_view(test.coupled_view);
  Kokkos::deep_copy(host_coupled_view, test.coupled_view);

  for (int i = 0; i < test.number_of_faces; ++i) {
    for (int j = 0; j < test.num_points; ++j) {
      for (int k = 0; k < test.num_points; ++k) {
        EXPECT_EQ(host_self_view(i, j, k), 3)
            << "Self intersection multiple iterations failed at face " << i
            << " point (" << j << "," << k << ") "
            << "for test: " << test.name;

        EXPECT_EQ(host_coupled_view(i, j, k), 3)
            << "Coupled intersection multiple iterations failed at face " << i
            << " point (" << j << "," << k << ") "
            << "for test: " << test.name;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    FaceIntersectionIteratorTests, FaceIntersectionIteratorTest,
    ::testing::Values(
        FaceIntersectionTestParams{ 10, "SmallIntersectionFaceValues" },
        FaceIntersectionTestParams{ 100, "MediumIntersectionFaceValues" },
        FaceIntersectionTestParams{ 1000, "LargeIntersectionFaceValues" },
        FaceIntersectionTestParams{ 16,
                                    "ExactChunkSizeIntersectionFaceValues" }));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
