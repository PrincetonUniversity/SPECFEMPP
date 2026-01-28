#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

/**
 * @brief Helper function to compute element coordinates from edge type and
 * local edge point index
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

class AssemblyEdgeViewTest : public ::testing::Test {
protected:
  using EdgeView = specfem::assembly::EdgeView<Kokkos::DefaultExecutionSpace>;
  using Edge = specfem::assembly::Edge<Kokkos::DefaultExecutionSpace>;

  static constexpr int num_edges = 4;
  static constexpr int num_points = 5;
};

TEST_F(AssemblyEdgeViewTest, DefaultConstructor) {
  EdgeView view;
  EXPECT_EQ(view.n_edges, 0);
  EXPECT_EQ(view.n_points, 0);
}

TEST_F(AssemblyEdgeViewTest, AllocatingConstructor) {
  EdgeView view("test_edges", num_edges, num_points);

  EXPECT_EQ(view.n_edges, num_edges);
  EXPECT_EQ(view.n_points, num_points);

  // Check view dimensions
  EXPECT_EQ(view.element_index.extent(0), num_edges);
  EXPECT_EQ(view.edge_index.extent(0), num_edges);
  EXPECT_EQ(view.edge_types.extent(0), num_edges);
  EXPECT_EQ(view.iz.extent(0), num_edges);
  EXPECT_EQ(view.iz.extent(1), num_points);
  EXPECT_EQ(view.ix.extent(0), num_edges);
  EXPECT_EQ(view.ix.extent(1), num_points);
}

TEST_F(AssemblyEdgeViewTest, SingleEdgeAccess) {
  EdgeView view("test_edges", num_edges, num_points);

  // Initialize edge data on device
  constexpr auto bottom = specfem::mesh_entity::dim2::type::bottom;

  Kokkos::parallel_for(
      "initialize_edges",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_edges),
      KOKKOS_LAMBDA(const int i) {
        view.element_index(i) = i * 10;
        view.edge_index(i) = i;
        view.edge_types(i) = bottom;
        for (int j = 0; j < num_points; ++j) {
          int iz_val, ix_val;
          get_edge_coordinates(bottom, j, num_points, iz_val, ix_val);
          view.iz(i, j) = iz_val;
          view.ix(i, j) = ix_val;
        }
      });
  Kokkos::fence();

  // Test single edge access
  Kokkos::View<int[3], Kokkos::DefaultExecutionSpace> results("results");

  Kokkos::parallel_for(
      "test_single_edge", 1, KOKKOS_LAMBDA(const int) {
        Edge edge = view(1);
        results(0) = edge.n_points;
        results(1) = edge.element_index;
        results(2) = edge.edge_index;
      });
  Kokkos::fence();

  auto host_results = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(host_results, results);

  EXPECT_EQ(host_results(0), num_points);
  EXPECT_EQ(host_results(1), 10); // element_index for edge 1
  EXPECT_EQ(host_results(2), 1);  // edge_index for edge 1
}

TEST_F(AssemblyEdgeViewTest, RangeAccess) {
  EdgeView view("test_edges", num_edges, num_points);

  // Initialize edge data
  constexpr auto top = specfem::mesh_entity::dim2::type::top;

  Kokkos::parallel_for(
      "initialize_edges",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_edges),
      KOKKOS_LAMBDA(const int i) {
        view.element_index(i) = i * 100;
        view.edge_index(i) = i;
        view.edge_types(i) = top;
        for (int j = 0; j < num_points; ++j) {
          int iz_val, ix_val;
          get_edge_coordinates(top, j, num_points, iz_val, ix_val);
          view.iz(i, j) = iz_val;
          view.ix(i, j) = ix_val;
        }
      });
  Kokkos::fence();

  // Test range access [1, 3) should give 2 edges
  Kokkos::View<int[4], Kokkos::DefaultExecutionSpace> results("results");

  Kokkos::parallel_for(
      "test_range_access", 1, KOKKOS_LAMBDA(const int) {
        EdgeView subview = view(Kokkos::make_pair(1, 3));
        results(0) = subview.n_edges;
        results(1) = subview.n_points;
        results(2) = subview.element_index(0); // First edge in subview
        results(3) = subview.element_index(1); // Second edge in subview
      });
  Kokkos::fence();

  auto host_results = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(host_results, results);

  EXPECT_EQ(host_results(0), 2);   // 2 edges in range [1, 3)
  EXPECT_EQ(host_results(1), num_points);
  EXPECT_EQ(host_results(2), 100); // element_index for original edge 1
  EXPECT_EQ(host_results(3), 200); // element_index for original edge 2
}

class AssemblyEdgeTest : public ::testing::Test {
protected:
  using Edge = specfem::assembly::Edge<Kokkos::DefaultExecutionSpace>;
  using IndexView =
      Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::DefaultExecutionSpace>;

  static constexpr int num_points = 5;
};

TEST_F(AssemblyEdgeTest, OperatorPointAccess) {
  // Create index views
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> iz_storage("iz",
                                                                 num_points);
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> ix_storage("ix",
                                                                 num_points);

  constexpr auto left = specfem::mesh_entity::dim2::type::left;

  // Initialize quadrature point coordinates
  Kokkos::parallel_for(
      "init_coords", num_points, KOKKOS_LAMBDA(const int j) {
        int iz_val, ix_val;
        get_edge_coordinates(left, j, num_points, iz_val, ix_val);
        iz_storage(j) = iz_val;
        ix_storage(j) = ix_val;
      });
  Kokkos::fence();

  // Test Edge operator()
  using EdgeIndex = specfem::point::edge_index<specfem::dimension::type::dim2>;
  Kokkos::View<int[6], Kokkos::DefaultExecutionSpace> results("results");

  Kokkos::parallel_for(
      "test_edge_operator", 1, KOKKOS_LAMBDA(const int) {
        // Create strided subviews matching Edge::IndexView type
        auto iz_strided = Kokkos::subview(iz_storage, Kokkos::ALL());
        auto ix_strided = Kokkos::subview(ix_storage, Kokkos::ALL());

        Edge edge(num_points, 42, 3, left, iz_strided, ix_strided);

        EdgeIndex idx = edge(2); // Access point 2

        results(0) = idx.ispec;
        results(1) = idx.iedge;
        results(2) = idx.ipoint;
        results(3) = idx.iz;
        results(4) = idx.ix;
        results(5) = static_cast<int>(idx.edge_type);
      });
  Kokkos::fence();

  auto host_results = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(host_results, results);

  EXPECT_EQ(host_results(0), 42); // element_index
  EXPECT_EQ(host_results(1), 3);  // edge_index
  EXPECT_EQ(host_results(2), 2);  // ipoint
  EXPECT_EQ(host_results(3), 2);  // iz for left edge, point 2
  EXPECT_EQ(host_results(4), 0);  // ix for left edge (always 0)
  EXPECT_EQ(host_results(5), static_cast<int>(left));
}

TEST_F(AssemblyEdgeTest, AllEdgeTypes) {
  // Test all four 2D edge types
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> iz_storage("iz",
                                                                 num_points);
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> ix_storage("ix",
                                                                 num_points);

  using EdgeIndex = specfem::point::edge_index<specfem::dimension::type::dim2>;

  // Test bottom edge
  {
    constexpr auto edge_type = specfem::mesh_entity::dim2::type::bottom;
    Kokkos::parallel_for(
        "init_bottom", num_points, KOKKOS_LAMBDA(const int j) {
          int iz_val, ix_val;
          get_edge_coordinates(edge_type, j, num_points, iz_val, ix_val);
          iz_storage(j) = iz_val;
          ix_storage(j) = ix_val;
        });
    Kokkos::fence();

    Kokkos::View<int[2], Kokkos::DefaultExecutionSpace> results("results");
    Kokkos::parallel_for(
        "test_bottom", 1, KOKKOS_LAMBDA(const int) {
          auto iz_strided = Kokkos::subview(iz_storage, Kokkos::ALL());
          auto ix_strided = Kokkos::subview(ix_storage, Kokkos::ALL());
          Edge edge(num_points, 0, 0, edge_type, iz_strided, ix_strided);
          EdgeIndex idx = edge(3);
          results(0) = idx.iz;
          results(1) = idx.ix;
        });
    Kokkos::fence();

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);
    EXPECT_EQ(host_results(0), 0); // iz = 0 for bottom
    EXPECT_EQ(host_results(1), 3); // ix = ipoint for bottom
  }

  // Test top edge
  {
    constexpr auto edge_type = specfem::mesh_entity::dim2::type::top;
    Kokkos::parallel_for(
        "init_top", num_points, KOKKOS_LAMBDA(const int j) {
          int iz_val, ix_val;
          get_edge_coordinates(edge_type, j, num_points, iz_val, ix_val);
          iz_storage(j) = iz_val;
          ix_storage(j) = ix_val;
        });
    Kokkos::fence();

    Kokkos::View<int[2], Kokkos::DefaultExecutionSpace> results("results");
    Kokkos::parallel_for(
        "test_top", 1, KOKKOS_LAMBDA(const int) {
          auto iz_strided = Kokkos::subview(iz_storage, Kokkos::ALL());
          auto ix_strided = Kokkos::subview(ix_storage, Kokkos::ALL());
          Edge edge(num_points, 0, 0, edge_type, iz_strided, ix_strided);
          EdgeIndex idx = edge(3);
          results(0) = idx.iz;
          results(1) = idx.ix;
        });
    Kokkos::fence();

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);
    EXPECT_EQ(host_results(0), num_points - 1); // iz = ngll-1 for top
    EXPECT_EQ(host_results(1), 3);              // ix = ipoint for top
  }

  // Test right edge
  {
    constexpr auto edge_type = specfem::mesh_entity::dim2::type::right;
    Kokkos::parallel_for(
        "init_right", num_points, KOKKOS_LAMBDA(const int j) {
          int iz_val, ix_val;
          get_edge_coordinates(edge_type, j, num_points, iz_val, ix_val);
          iz_storage(j) = iz_val;
          ix_storage(j) = ix_val;
        });
    Kokkos::fence();

    Kokkos::View<int[2], Kokkos::DefaultExecutionSpace> results("results");
    Kokkos::parallel_for(
        "test_right", 1, KOKKOS_LAMBDA(const int) {
          auto iz_strided = Kokkos::subview(iz_storage, Kokkos::ALL());
          auto ix_strided = Kokkos::subview(ix_storage, Kokkos::ALL());
          Edge edge(num_points, 0, 0, edge_type, iz_strided, ix_strided);
          EdgeIndex idx = edge(3);
          results(0) = idx.iz;
          results(1) = idx.ix;
        });
    Kokkos::fence();

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);
    EXPECT_EQ(host_results(0), 3);              // iz = ipoint for right
    EXPECT_EQ(host_results(1), num_points - 1); // ix = ngll-1 for right
  }
}

class AssemblyEdgeViewHostMirrorTest : public ::testing::Test {
protected:
  using EdgeView = specfem::assembly::EdgeView<Kokkos::DefaultExecutionSpace>;

  static constexpr int num_edges = 3;
  static constexpr int num_points = 4;
};

TEST_F(AssemblyEdgeViewHostMirrorTest, HostMirrorType) {
  // Verify HostMirror type is correctly defined
  using HostMirror = typename EdgeView::HostMirror;

  HostMirror host_view("host_edges", num_edges, num_points);

  EXPECT_EQ(host_view.n_edges, num_edges);
  EXPECT_EQ(host_view.n_points, num_points);

  // Initialize on host
  for (int i = 0; i < num_edges; ++i) {
    host_view.element_index(i) = i;
    host_view.edge_index(i) = i * 2;
    host_view.edge_types(i) = specfem::mesh_entity::dim2::type::bottom;
    for (int j = 0; j < num_points; ++j) {
      host_view.iz(i, j) = 0;
      host_view.ix(i, j) = j;
    }
  }

  // Verify values
  EXPECT_EQ(host_view.element_index(1), 1);
  EXPECT_EQ(host_view.edge_index(2), 4);
  EXPECT_EQ(host_view.ix(0, 2), 2);
}

