#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include "enumerations/interface.hpp"
#include "execution/chunked_edge_iterator.hpp"
#include "execution/chunked_intersection_iterator.hpp"
#include "execution/for_all.hpp"
#include "parallel_configuration/chunk_edge_config.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

// Base fixture for common functionality
class ChunkedIteratorTestBase {
public:
  using ParallelConfig = specfem::parallel_config::default_chunk_edge_config<
      specfem::dimension::type::dim2, Kokkos::DefaultExecutionSpace>;

  constexpr static int num_points = 5;
  using StorageViewType =
      Kokkos::View<int *[num_points], Kokkos::DefaultExecutionSpace>;
  using EdgesViewType =
      Kokkos::View<specfem::mesh_entity::edge *, Kokkos::DefaultExecutionSpace>;
};

// Test parameter structs (no Kokkos views here)
struct EdgeIteratorTestParams {
  std::size_t number_of_edges;
  std::string name;

  EdgeIteratorTestParams(std::size_t n, const char *test_name)
      : number_of_edges(n), name(test_name) {}
};

std::ostream &operator<<(std::ostream &os,
                         const EdgeIteratorTestParams &params) {
  os << params.name;
  return os;
}

struct IntersectionIteratorTestParams {
  std::size_t number_of_edges;
  std::string name;

  IntersectionIteratorTestParams(std::size_t n, const char *test_name)
      : number_of_edges(n), name(test_name) {}
};

std::ostream &operator<<(std::ostream &os,
                         const IntersectionIteratorTestParams &params) {
  os << params.name;
  return os;
}

// Fixture specifically for Edge Iterator tests
class EdgeIterator : public ChunkedIteratorTestBase {
public:
  StorageViewType view;
  EdgesViewType edges;
  std::string name;
  int number_of_edges;

  EdgeIterator(const EdgeIteratorTestParams &params)
      : view("view", params.number_of_edges),
        edges("edges", params.number_of_edges), name(params.name),
        number_of_edges(params.number_of_edges) {

    this->reset();
    Kokkos::fence();
  }

  void run() const {
    specfem::execution::ChunkedEdgeIterator iterator(
        ParallelConfig(), this->edges, this->num_points);
    specfem::execution::for_all(
        "test_chunked_edge_iterator", iterator,
        KOKKOS_CLASS_LAMBDA(
            const typename decltype(iterator)::base_index_type &index) {
          Kokkos::atomic_add(&view(index.ispec, index.ipoint), 1);
        });

    Kokkos::fence();
  }

  void check() const {
    auto host_view = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(host_view, view);

    for (std::size_t i = 0; i < host_view.extent(0); ++i) {
      for (int j = 0; j < this->num_points; ++j) {
        EXPECT_EQ(host_view(i, j), 1)
            << "Edge iterator failed at (" << i << "," << j << ") "
            << "for test: " << name;
      }
    }
  }

  void reset() const {
    Kokkos::parallel_for(
        "initialize_edges",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_edges),
        KOKKOS_CLASS_LAMBDA(const int i) {
          for (int j = 0; j < num_points; ++j)
            view(i, j) = 0;
          edges(i) = specfem::mesh_entity::edge(
              static_cast<int>(i), specfem::mesh_entity::type::top);
        });
  }
};

// Fixture specifically for Intersection Iterator tests
class IntersectionIterator : public ChunkedIteratorTestBase {
public:
  StorageViewType self_view;
  StorageViewType coupled_view;
  EdgesViewType edges;
  EdgesViewType intersection_edges;
  std::string name;
  int number_of_edges;

  IntersectionIterator(const IntersectionIteratorTestParams &params)
      : self_view("self_view", params.number_of_edges),
        coupled_view("coupled_view", params.number_of_edges),
        edges("edges", params.number_of_edges),
        intersection_edges("intersection_edges", params.number_of_edges),
        name(params.name), number_of_edges(params.number_of_edges) {

    this->reset();
    Kokkos::fence();
  }

  void run() const {
    specfem::execution::ChunkedIntersectionIterator iterator(
        ParallelConfig(), edges, intersection_edges, num_points);
    specfem::execution::for_all(
        "test_chunked_intersection_edge_iterator", iterator,
        KOKKOS_CLASS_LAMBDA(
            const typename decltype(iterator)::base_index_type &index) {
          const auto self_index = index.self_index;
          const auto coupled_index = index.coupled_index;
          Kokkos::atomic_add(&self_view(self_index.ispec, self_index.ipoint),
                             1);
          Kokkos::atomic_add(
              &coupled_view(coupled_index.ispec, coupled_index.ipoint), 1);
        });
    Kokkos::fence();
  }

  void check() const {
    auto host_self_view = Kokkos::create_mirror_view(self_view);
    Kokkos::deep_copy(host_self_view, self_view);
    auto host_coupled_view = Kokkos::create_mirror_view(coupled_view);
    Kokkos::deep_copy(host_coupled_view, coupled_view);

    for (std::size_t i = 0; i < number_of_edges; ++i) {
      for (int j = 0; j < num_points; ++j) {
        EXPECT_EQ(host_self_view(i, j), 1)
            << "Intersection iterator failed at (" << i << "," << j << ") "
            << "expected: 1 for test: " << name;

        EXPECT_EQ(host_coupled_view(i, j), 1)
            << "Intersection iterator failed at (" << i << "," << j << ") "
            << "expected: 1 for test: " << name;
      }
    }
  }

  void reset() const {
    Kokkos::parallel_for(
        "initialize_intersection_edges",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_edges),
        KOKKOS_CLASS_LAMBDA(const int i) {
          for (int j = 0; j < num_points; ++j) {
            self_view(i, j) = 0;
            coupled_view(i, j) = 0;
          }
          edges(i) = specfem::mesh_entity::edge(
              static_cast<int>(i), specfem::mesh_entity::type::top);
          intersection_edges(i) = specfem::mesh_entity::edge(
              static_cast<int>(number_of_edges - i - 1),
              specfem::mesh_entity::type::bottom);
        });
  }
};

// Value parameterized tests
class EdgeIteratorTest
    : public ::testing::TestWithParam<EdgeIteratorTestParams> {
protected:
  void SetUp() override {
    // Common setup if needed
  }

  void TearDown() override {
    // Common cleanup if needed
  }
};

class IntersectionIteratorTest
    : public ::testing::TestWithParam<IntersectionIteratorTestParams> {
protected:
  void SetUp() override {
    // Common setup if needed
  }

  void TearDown() override {
    // Common cleanup if needed
  }
};

TEST_P(EdgeIteratorTest, VisitAllPoints) {
  const EdgeIterator test(GetParam());
  test.reset();
  Kokkos::fence();
  test.run();
  test.check();
}

TEST_P(IntersectionIteratorTest, VisitAllPoints) {
  const IntersectionIterator test(GetParam());
  test.reset();
  Kokkos::fence();
  test.run();
  test.check();
}

INSTANTIATE_TEST_SUITE_P(
    EdgeIteratorTests, EdgeIteratorTest,
    ::testing::Values(EdgeIteratorTestParams{ 10, "SmallEdgeValues" },
                      EdgeIteratorTestParams{ 1000, "LargeEdgeValues" },
                      EdgeIteratorTestParams{ 10000, "VeryLargeEdgeValues" },
                      EdgeIteratorTestParams{ 1024,
                                              "ExactChunkSizeEdgeValues" }));

INSTANTIATE_TEST_SUITE_P(
    IntersectionIteratorTests, IntersectionIteratorTest,
    ::testing::Values(
        IntersectionIteratorTestParams{ 10, "SmallIntersectionEdgeValues" },
        IntersectionIteratorTestParams{ 1000, "LargeIntersectionEdgeValues" },
        IntersectionIteratorTestParams{ 10000,
                                        "VeryLargeIntersectionEdgeValues" },
        IntersectionIteratorTestParams{
            1024, "ExactChunkSizeIntersectionEdgeValues" }));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
