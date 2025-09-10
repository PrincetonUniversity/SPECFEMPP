#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include <boost/graph/filtered_graph.hpp>
#include <gtest/gtest.h>

#include "algorithms/locate_point.hpp"
#include "specfem/assembly/nonconforming_interfaces/dim2/compute_intersection.hpp"

#include "nonconforming/interfacial_assembly/interfacial_assembly.hpp"
#include "specfem/point/coordinates.hpp"

struct interval_unions {
  std::vector<std::pair<type_real, type_real> > intervals;

  interval_unions() = default;

  /**
   * @brief insert the interval [a,b].
   * @return true if any intervals had nonzero overlap
   */
  bool insert(type_real a, type_real b) {
    if (a > b)
      std::swap(a, b);
    intervals.push_back({ a, b });
    return merge_intervals();
  }

  /**
   * @brief Merge overlapping intervals
   * @return true if any intervals had nonzero overlap
   */
  bool merge_intervals() {
    if (intervals.empty())
      return false;
    std::sort(intervals.begin(), intervals.end());
    std::vector<std::pair<type_real, type_real> > merged;
    merged.push_back(intervals[0]);

    bool had_overlap = false;

    // combine overlapping intervals
    for (size_t i = 1; i < intervals.size(); ++i) {
      if (intervals[i].first <= merged.back().second + 1e-3) {
        if (intervals[i].first + 1e-3 < merged.back().second) {
          had_overlap = true;
        }
        merged.back().second =
            std::max(merged.back().second, intervals[i].second);
      } else {
        merged.push_back(intervals[i]);
      }
    }
    intervals = std::move(merged);
    return had_overlap;
  }

  std::string to_string() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < intervals.size(); ++i) {
      oss << "(" << intervals[i].first << ", " << intervals[i].second << ")";
      if (i + 1 < intervals.size())
        oss << " ⊔ ";
    }
    oss << "]";
    return oss.str();
  }

  /**
   * @brief Check if the union of intervals spans from -1 to 1
   */
  bool is_full() const {
    return intervals.size() == 1 && intervals[0].first <= -1.0 + 1e-3 &&
           intervals[0].second >= 1.0 - 1e-3;
  }
};

/**
 * @brief Test that the intersections computed cover the entire interface
 *
 * This function computes the intersections for all edges between two sides of
 * a nonconforming interface, and checks:
 *  1) that the union of all intersections on each side covers the entire
 *     edge (i.e., from -1 to 1 in local coordinates).
 *  2) that there are no overlapping intersections on each side.
 *
 * @param interface_config Configuration of the nonconforming interface
 * @param assembly Assembly object containing the mesh
 */
void test_intersection_completeness(
    const specfem::testing::interfacial_assembly_config &interface_config,
    const specfem::assembly::assembly<specfem::dimension::type::dim2>
        &assembly) {

  const int &nelem_side1 = interface_config.nelem_side1;
  const int &nelem_side2 = interface_config.nelem_side2;

  const auto &mesh = assembly.mesh;

  const auto &graph = mesh.graph();

  auto mortar_quadrature =
      specfem::quadrature::gll::gll(0, 0, interface_config.ngll);

  std::vector<interval_unions> side1_intervals_bottom_to_top(nelem_side1);
  std::vector<interval_unions> side2_intervals_bottom_to_top(nelem_side2);
  std::vector<interval_unions> side1_intervals_top_to_bottom(nelem_side1);
  std::vector<interval_unions> side2_intervals_top_to_bottom(nelem_side2);

  for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
    const int ispec = mesh.compute_to_mesh(boost::source(edge, graph));
    const int jspec = mesh.compute_to_mesh(boost::target(edge, graph));

    if (ispec < nelem_side1 && jspec >= nelem_side1) {
      // bottom to top
      auto intersections =
          specfem::assembly::nonconforming_interfaces::compute_intersection(
              mesh, edge, mortar_quadrature);
      // compute min and max of local coord on each side
      type_real lo_side1 = 1.0;
      type_real hi_side1 = -1.0;
      type_real lo_side2 = 1.0;
      type_real hi_side2 = -1.0;
      for (const auto &[a, b] : intersections) {
        lo_side1 = std::min(lo_side1, a);
        hi_side1 = std::max(hi_side1, a);
        lo_side2 = std::min(lo_side2, b);
        hi_side2 = std::max(hi_side2, b);
      }
      if (side1_intervals_bottom_to_top[ispec].insert(lo_side1, hi_side1)) {
        std::ostringstream oss;
        oss << "Overlapping intervals on side 1 element " << ispec
            << " (bottom to top)\n";
        throw std::runtime_error("Overlapping intervals on side 1");
      }
      if (side2_intervals_bottom_to_top[jspec - nelem_side1].insert(lo_side2,
                                                                    hi_side2)) {
        throw std::runtime_error("Overlapping intervals on side 2");
      }
    } else if (jspec < nelem_side1 && ispec >= nelem_side1) {
      // top to bottom
      auto intersections =
          specfem::assembly::nonconforming_interfaces::compute_intersection(
              mesh, edge, mortar_quadrature);
      // compute min and max of local coord on each side
      type_real lo_side1 = 1.0;
      type_real hi_side1 = -1.0;
      type_real lo_side2 = 1.0;
      type_real hi_side2 = -1.0;
      for (const auto &[a, b] : intersections) {
        lo_side1 = std::min(lo_side1, b);
        hi_side1 = std::max(hi_side1, b);
        lo_side2 = std::min(lo_side2, a);
        hi_side2 = std::max(hi_side2, a);
      }
      if (side1_intervals_top_to_bottom[jspec].insert(lo_side1, hi_side1)) {
        throw std::runtime_error("Overlapping intervals on side 1");
      }
      if (side2_intervals_top_to_bottom[ispec - nelem_side1].insert(lo_side2,
                                                                    hi_side2)) {
        throw std::runtime_error("Overlapping intervals on side 2");
      }
    }
  }

  for (int ielem = 0; ielem < nelem_side1; ++ielem) {
    if (!side1_intervals_bottom_to_top[ielem].is_full()) {
      std::ostringstream oss;
      oss << "Side 1 element " << ielem
          << " does not have full intersection coverage (bottom to top)\n"
          << "Intervals: " << side1_intervals_bottom_to_top[ielem].to_string()
          << "\n";
      throw std::runtime_error(oss.str());
    }
    if (!side1_intervals_top_to_bottom[ielem].is_full()) {
      std::ostringstream oss;
      oss << "Side 1 element " << ielem
          << " does not have full intersection coverage (top to bottom)\n"
          << "Intervals: " << side1_intervals_top_to_bottom[ielem].to_string()
          << "\n";
      ;
      throw std::runtime_error(oss.str());
    }
  }
  for (int ielem = 0; ielem < nelem_side2; ++ielem) {
    if (!side2_intervals_bottom_to_top[ielem].is_full()) {
      std::ostringstream oss;
      oss << "Side 2 element " << ielem
          << " does not have full intersection coverage (bottom to top)\n"
          << "Intervals: " << side2_intervals_bottom_to_top[ielem].to_string()
          << "\n";
      ;
      throw std::runtime_error(oss.str());
    }
    if (!side2_intervals_top_to_bottom[ielem].is_full()) {
      std::ostringstream oss;
      oss << "Side 2 element " << ielem
          << " does not have full intersection coverage (top to bottom)\n"
          << "Intervals: " << side2_intervals_top_to_bottom[ielem].to_string()
          << "\n";
      ;
      throw std::runtime_error(oss.str());
    }
  }
}

/**
 * @brief Test that the intersections computed have pointwise agreement
 *
 * This function computes the intersections for all edges between two sides of
 * a nonconforming interface, and checks if the local coordinates
 * recovered from compute_intersection on both sides have the same global
 * coordinate.
 *
 * @param interface_config Configuration of the nonconforming interface
 * @param assembly Assembly object containing the mesh
 */
void test_intersection_pointwise_agreement(
    const specfem::testing::interfacial_assembly_config &interface_config,
    const specfem::assembly::assembly<specfem::dimension::type::dim2>
        &assembly) {
  const type_real eps = 1e-5 * interface_config.interface_length;
  const int &nelem_side1 = interface_config.nelem_side1;
  const int &nelem_side2 = interface_config.nelem_side2;
  const auto &mesh = assembly.mesh;
  const auto &graph = mesh.graph();

  // Filter out strongly conforming connections
  auto filter = [&graph](const auto &edge) {
    return graph[edge].connection == specfem::connections::type::nonconforming;
  };

  // Create a filtered graph view
  const auto fg = boost::make_filtered_graph(graph, filter);

  auto mortar_quadrature =
      specfem::quadrature::gll::gll(0, 0, interface_config.ngll);

  for (auto edge : boost::make_iterator_range(boost::edges(fg))) {
    const int ispec_assem = boost::source(edge, fg);
    const int jspec_assem = boost::target(edge, fg);
    const int ispec = mesh.compute_to_mesh(ispec_assem);
    const int jspec = mesh.compute_to_mesh(jspec_assem);

    const specfem::mesh_entity::type iorientation = fg[edge].orientation;
    const auto [edge_inv, exists] = boost::edge(jspec_assem, ispec_assem, fg);
    if (!exists) {
      std::ostringstream oss;
      oss << "Non-symmetric adjacency graph! Edge " << ispec << " -> " << jspec
          << " exists, but not the other way around.";
      throw std::runtime_error(oss.str());
    }
    const specfem::mesh_entity::type jorientation = fg[edge_inv].orientation;

    auto intersections =
        specfem::assembly::nonconforming_interfaces::compute_intersection(
            mesh, edge, mortar_quadrature);
    // check pointwise agreement
    for (int i = 0; i < mortar_quadrature.get_N(); ++i) {
      const auto [a, b] = intersections[i];

      // map to global coordinates and check agreement
      auto p1 = specfem::algorithms::locate_point_on_edge(a, mesh, ispec,
                                                          iorientation);
      auto p2 = specfem::algorithms::locate_point_on_edge(b, mesh, jspec,
                                                          jorientation);
      if (specfem::point::distance(p1, p2) > eps) {
        std::ostringstream oss;
        oss << "Pointwise agreement failed for edge " << ispec << " ("
            << specfem::mesh_entity::to_string(iorientation) << ") <-> "
            << jspec << " (" << specfem::mesh_entity::to_string(jorientation)
            << ") for quadrature point " << i << ":\n"
            << " - Point on " << ispec << " @ local coord " << a << ": ("
            << p1.x << ", " << p1.z << ")\n"
            << " - Point on " << jspec << " @ local coord " << b << ": ("
            << p2.x << ", " << p2.z << ")\n"
            << " - Distance: " << std::scientific
            << specfem::point::distance(p1, p2) << "\n";
        throw std::runtime_error(oss.str());
      }
    }
  }
}

using INTERFACIAL_ASSEMBLY_FIXTURE =
    specfem::testing::INTERFACIAL_ASSEMBLY_FIXTURE;

TEST_F(INTERFACIAL_ASSEMBLY_FIXTURE, IntersectionCompleteness) {
  for (const auto &[interface_config, assembly] : *this) {
    try {
      test_intersection_completeness(interface_config, assembly);
    } catch (const std::exception &e) {
      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " Test: IntersectionCompleteness\n"
             << " - Interface: length = " << interface_config.interface_length
             << ", ngll = " << interface_config.ngll << "\n"
             << "     side 1: nelem = " << interface_config.nelem_side1 << "\n"
             << "     side 1: nelem = " << interface_config.nelem_side2 << "\n"
             << " - Exception thrown: " << e.what() << "\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
  }
}

TEST_F(INTERFACIAL_ASSEMBLY_FIXTURE, IntersectionPointwiseAgreement) {
  for (const auto &[interface_config, assembly] : *this) {
    try {
      test_intersection_pointwise_agreement(interface_config, assembly);
    } catch (const std::exception &e) {
      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " Test: IntersectionPointwiseAgreement\n"
             << " - Interface: length = " << interface_config.interface_length
             << ", ngll = " << interface_config.ngll << "\n"
             << "     side 1: nelem = " << interface_config.nelem_side1 << "\n"
             << "     side 1: nelem = " << interface_config.nelem_side2 << "\n"
             << " - Exception thrown: " << e.what() << "\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
  }
}
