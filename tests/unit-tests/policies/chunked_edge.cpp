#include "../SPECFEM_Environment.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"
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
class ChunkedIteratorTestBase {
public:
  using ParallelConfig =
      specfem::parallel_configuration::default_chunk_edge_config<
          specfem::element::dimension_tag::dim2, Kokkos::DefaultExecutionSpace>;

  constexpr static int num_points = 5;
  // Storage view indexed by [edge][ipoint]
  using StorageViewType = Kokkos::View<int **, Kokkos::DefaultExecutionSpace>;
  using EdgesViewType =
      specfem::assembly::EdgeView<Kokkos::DefaultExecutionSpace>;
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
  specfem::mesh_entity::element<specfem::element::dimension_tag::dim2> elem;

  EdgeIterator(const EdgeIteratorTestParams &params)
      : view("view", params.number_of_edges, num_points),
        edges("edges", params.number_of_edges, num_points), name(params.name),
        number_of_edges(params.number_of_edges), elem(num_points) {

    this->reset();
    Kokkos::fence();
  }

  void run() const {
    specfem::execution::ChunkedEdgeIterator iterator(ParallelConfig(),
                                                     this->edges);
    specfem::execution::for_all(
        "test_chunked_edge_iterator", iterator,
        KOKKOS_CLASS_LAMBDA(const typename decltype(
            iterator)::base_index_type &iterator_index) {
          const auto index = iterator_index.get_index();
          Kokkos::atomic_add(&view(index.iedge, index.ipoint), 1);
        });

    Kokkos::fence();
  }

  void check() const {
    auto host_view = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(host_view, view);

    for (int i = 0; i < number_of_edges; ++i) {
      for (int j = 0; j < num_points; ++j) {
        EXPECT_EQ(host_view(i, j), 1)
            << "Edge iterator failed at (" << i << "," << j << ") "
            << "for test: " << name;
      }
    }
  }

  void reset() const {
    // Initialize storage view to zeros
    Kokkos::parallel_for(
        "initialize_storage",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_edges),
        KOKKOS_CLASS_LAMBDA(const int i) {
          for (int j = 0; j < num_points; ++j) {
            view(i, j) = 0;
          }
        });

    // Initialize edges view - cycle through edge types
    constexpr auto bottom = specfem::mesh_entity::dim2::type::bottom;
    constexpr auto top = specfem::mesh_entity::dim2::type::top;
    constexpr auto left = specfem::mesh_entity::dim2::type::left;
    constexpr auto right = specfem::mesh_entity::dim2::type::right;

    Kokkos::parallel_for(
        "initialize_edges",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_edges),
        KOKKOS_CLASS_LAMBDA(const int i) {
          edges.element_index(i) = static_cast<int>(i);
          edges.edge_index(i) = static_cast<int>(i);
          // Cycle through edge types
          switch (i % 4) {
          case 0:
            edges.edge_types(i) = bottom;
            break;
          case 1:
            edges.edge_types(i) = top;
            break;
          case 2:
            edges.edge_types(i) = left;
            break;
          case 3:
            edges.edge_types(i) = right;
            break;
          }

          // Set up quadrature point coordinates based on edge type
          for (int ipoint = 0; ipoint < num_points; ++ipoint) {
            int iz_val, ix_val;
            elem.get_edge_coordinates(edges.edge_types(i), ipoint, iz_val,
                                      ix_val);
            edges.iz(i, ipoint) = iz_val;
            edges.ix(i, ipoint) = ix_val;
          }
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
  specfem::mesh_entity::element<specfem::element::dimension_tag::dim2> elem;

  IntersectionIterator(const IntersectionIteratorTestParams &params)
      : self_view("self_view", params.number_of_edges, num_points),
        coupled_view("coupled_view", params.number_of_edges, num_points),
        edges("edges", params.number_of_edges, num_points),
        intersection_edges("intersection_edges", params.number_of_edges,
                           num_points),
        name(params.name), number_of_edges(params.number_of_edges),
        elem(num_points) {

    this->reset();
    Kokkos::fence();
  }

  void run() const {
    specfem::execution::ChunkedIntersectionIterator iterator(
        ParallelConfig(), edges, intersection_edges);
    specfem::execution::for_all(
        "test_chunked_intersection_edge_iterator", iterator,
        KOKKOS_CLASS_LAMBDA(const typename decltype(
            iterator)::base_index_type &iterator_index) {
          const auto index = iterator_index.get_index();
          const auto self_index = index.self_index;
          const auto coupled_index = index.coupled_index;
          Kokkos::atomic_add(&self_view(self_index.iedge, self_index.ipoint),
                             1);
          Kokkos::atomic_add(
              &coupled_view(coupled_index.iedge, coupled_index.ipoint), 1);
        });
    Kokkos::fence();
  }

  void check() const {
    auto host_self_view = Kokkos::create_mirror_view(self_view);
    Kokkos::deep_copy(host_self_view, self_view);
    auto host_coupled_view = Kokkos::create_mirror_view(coupled_view);
    Kokkos::deep_copy(host_coupled_view, coupled_view);

    for (int i = 0; i < number_of_edges; ++i) {
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
    // Initialize storage views to zeros
    Kokkos::parallel_for(
        "initialize_storage",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_edges),
        KOKKOS_CLASS_LAMBDA(const int i) {
          for (int j = 0; j < num_points; ++j) {
            self_view(i, j) = 0;
            coupled_view(i, j) = 0;
          }
        });

    // Initialize edges views
    constexpr auto top = specfem::mesh_entity::dim2::type::top;
    constexpr auto bottom = specfem::mesh_entity::dim2::type::bottom;

    Kokkos::parallel_for(
        "initialize_intersection_edges",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, number_of_edges),
        KOKKOS_CLASS_LAMBDA(const int i) {
          // Self edges
          edges.element_index(i) = static_cast<int>(i);
          edges.edge_index(i) = static_cast<int>(i);
          edges.edge_types(i) = top;

          // Coupled edges (reversed order)
          intersection_edges.element_index(i) =
              static_cast<int>(number_of_edges - i - 1);
          intersection_edges.edge_index(i) = static_cast<int>(i);
          intersection_edges.edge_types(i) = bottom;

          // Set up quadrature point coordinates
          for (int ipoint = 0; ipoint < num_points; ++ipoint) {
            int iz_val, ix_val;
            elem.get_edge_coordinates(top, ipoint, iz_val, ix_val);
            edges.iz(i, ipoint) = iz_val;
            edges.ix(i, ipoint) = ix_val;

            elem.get_edge_coordinates(bottom, ipoint, iz_val, ix_val);
            intersection_edges.iz(i, ipoint) = iz_val;
            intersection_edges.ix(i, ipoint) = ix_val;
          }
        });
  }
};

// Value parameterized tests
class EdgeIteratorTest
    : public ::testing::TestWithParam<EdgeIteratorTestParams> {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

class IntersectionIteratorTest
    : public ::testing::TestWithParam<IntersectionIteratorTestParams> {
protected:
  void SetUp() override {}
  void TearDown() override {}
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
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
