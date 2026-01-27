#include "../SPECFEM_Environment.hpp"
#include "enumerations/interface.hpp"
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

/**
 * @brief Helper function to compute element coordinates from edge type and
 * local edge point index
 *
 * Maps edge-local ipoint coordinate to element-local (iz, ix) coordinates
 * based on edge type.
 */
KOKKOS_INLINE_FUNCTION
void get_edge_coordinates(const specfem::mesh_entity::dim2::type edge_type,
                          const int ipoint, const int ngll, int &iz, int &ix) {
  switch (edge_type) {
  case specfem::mesh_entity::dim2::type::bottom:
    iz = 0;
    ix = ipoint;
    break;
  case specfem::mesh_entity::dim2::type::top:
    iz = ngll - 1;
    ix = ipoint;
    break;
  case specfem::mesh_entity::dim2::type::left:
    iz = ipoint;
    ix = 0;
    break;
  case specfem::mesh_entity::dim2::type::right:
    iz = ipoint;
    ix = ngll - 1;
    break;
  default:
    iz = -1;
    ix = -1;
    break;
  }
}

} // namespace

/**
 * @brief Individual 2D edge representation with quadrature point access
 *
 * This structure represents a single edge in a 2D spectral element mesh,
 * providing access to quadrature points on the edge for coupling
 * computations and boundary condition enforcement.
 *
 * @tparam ExecutionSpace Kokkos execution space (host or device)
 */
template <typename ExecutionSpace> struct Edge {
  int n_points;      ///< Number of quadrature points per edge
  int element_index; ///< Index of the spectral element containing this edge
  int edge_index;    ///< Global edge index
  specfem::mesh_entity::dim2::type edge_type; ///< Edge type
  using IndexView = Kokkos::View<int *, Kokkos::LayoutStride,
                                 ExecutionSpace>; ///< View for quadrature
                                                  ///< indices
  IndexView iz;                                   ///< Z-coordinate indices
  IndexView ix;                                   ///< X-coordinate indices

  KOKKOS_INLINE_FUNCTION
  Edge(const int n_points_, const int element_index_, const int edge_index_,
       const specfem::mesh_entity::dim2::type edge_type_, const IndexView &iz_,
       const IndexView &ix_)
      : n_points(n_points_), element_index(element_index_),
        edge_index(edge_index_), edge_type(edge_type_), iz(iz_), ix(ix_) {}

  /**
   * @brief Access quadrature point on the edge
   *
   * @param ipoint Point index along the edge (0 to n_points-1)
   * @return edge_index for the specified quadrature point
   */
  KOKKOS_INLINE_FUNCTION
  specfem::point::edge_index<specfem::dimension::type::dim2>
  operator()(const int ipoint) const {
    return { element_index, edge_index, ipoint, iz(ipoint), ix(ipoint),
             edge_type };
  }
};

/**
 * @brief Collection of 2D edges with parallel access capabilities
 *
 * This structure manages collections of edges for efficient parallel
 * processing of edge-based operations.
 *
 * @tparam ExecutionSpace Kokkos execution space (host or device)
 * @tparam Layout Memory layout for Kokkos views
 */
template <typename ExecutionSpace,
          typename Layout = typename ExecutionSpace::array_layout>
struct EdgeView {
  int n_edges;  ///< Number of edges in this view
  int n_points; ///< Number of quadrature points per edge
  using IndexView = Kokkos::View<int *, Layout, ExecutionSpace>;
  using QPView = Kokkos::View<int **, Layout, ExecutionSpace>;
  using EdgeTypeView =
      Kokkos::View<specfem::mesh_entity::dim2::type *, ExecutionSpace>;

  using HostMirror = std::conditional_t<
      std::is_same<typename ExecutionSpace::memory_space,
                   Kokkos::HostSpace>::value,
      EdgeView, EdgeView<Kokkos::DefaultHostExecutionSpace, Layout>>;

  EdgeView() : n_edges(0), n_points(0) {}

  EdgeView(const std::string &label, const int n_edges_, const int n_points_)
      : n_edges(n_edges_), n_points(n_points_),
        element_index(label + "_element_index", n_edges_),
        edge_index(label + "_edge_index", n_edges_),
        edge_types(label + "_edge_types", n_edges_),
        iz(label + "_iz", n_edges_, n_points_),
        ix(label + "_ix", n_edges_, n_points_) {}

  IndexView element_index;
  IndexView edge_index;
  EdgeTypeView edge_types;
  QPView iz;
  QPView ix;

  KOKKOS_INLINE_FUNCTION
  EdgeView(const int n_edges_, const int n_points_,
           const IndexView &element_index_, const IndexView &edge_index_,
           const EdgeTypeView &edge_types_, const QPView &iz_, const QPView &ix_)
      : n_edges(n_edges_), n_points(n_points_), element_index(element_index_),
        edge_index(edge_index_), edge_types(edge_types_), iz(iz_), ix(ix_) {}

  /**
   * @brief Access individual edge by index
   */
  KOKKOS_INLINE_FUNCTION
  Edge<ExecutionSpace> operator()(const int edge_id) const {
    return { n_points,
             element_index(edge_id),
             edge_index(edge_id),
             edge_types(edge_id),
             Kokkos::subview(iz, edge_id, Kokkos::ALL()),
             Kokkos::subview(ix, edge_id, Kokkos::ALL()) };
  }

  /**
   * @brief Access subrange of edges
   */
  KOKKOS_INLINE_FUNCTION
  EdgeView<ExecutionSpace>
  operator()(const Kokkos::pair<int, int> &edge_range) const {
    return { edge_range.second - edge_range.first,
             n_points,
             Kokkos::subview(element_index, edge_range),
             Kokkos::subview(edge_index, edge_range),
             Kokkos::subview(edge_types, edge_range),
             Kokkos::subview(iz, edge_range, Kokkos::ALL()),
             Kokkos::subview(ix, edge_range, Kokkos::ALL()) };
  }
};

// Base fixture for common functionality
class ChunkedIteratorTestBase {
public:
  using ParallelConfig =
      specfem::parallel_configuration::default_chunk_edge_config<
          specfem::dimension::type::dim2, Kokkos::DefaultExecutionSpace>;

  constexpr static int num_points = 5;
  // Storage view indexed by [edge][ipoint]
  using StorageViewType =
      Kokkos::View<int **, Kokkos::DefaultExecutionSpace>;
  using EdgesViewType = EdgeView<Kokkos::DefaultExecutionSpace>;
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
      : view("view", params.number_of_edges, num_points),
        edges("edges", params.number_of_edges, num_points), name(params.name),
        number_of_edges(params.number_of_edges) {

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
            get_edge_coordinates(edges.edge_types(i), ipoint, num_points,
                                 iz_val, ix_val);
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

  IntersectionIterator(const IntersectionIteratorTestParams &params)
      : self_view("self_view", params.number_of_edges, num_points),
        coupled_view("coupled_view", params.number_of_edges, num_points),
        edges("edges", params.number_of_edges, num_points),
        intersection_edges("intersection_edges", params.number_of_edges,
                           num_points),
        name(params.name), number_of_edges(params.number_of_edges) {

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
            get_edge_coordinates(top, ipoint, num_points, iz_val, ix_val);
            edges.iz(i, ipoint) = iz_val;
            edges.ix(i, ipoint) = ix_val;

            get_edge_coordinates(bottom, ipoint, num_points, iz_val, ix_val);
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
