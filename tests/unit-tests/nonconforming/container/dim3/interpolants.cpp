#include "specfem/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace specfem::nonconforming_test::interpolants_impl {

using assembly_type =
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>;
using meshing_type = specfem::element_coupling::interfacial_meshing_type;

/// Tolerance on the sum of one row of 1D Lagrange interpolants. A row is
/// evaluated from at most 8 GLL nodes, so a few ulps of accumulated rounding,
/// ~8 * 6e-8 for single precision, are expected; 1e-5 leaves a margin of
/// more than an order of magnitude without hiding a missing or doubled
/// polynomial (which changes the sum by O(1)).
constexpr double partition_of_unity_tol =
    std::is_same_v<type_real, float> ? 1e-5 : 1e-12;

/// Tolerance on the reconstructed node position, relative to the largest
/// corner-to-corner distance of the guest face. The faces of these fixtures
/// are flat, so the guest geometry is exactly reproduced by its degree-(N-1)
/// interpolation; what remains is the accuracy of the stored local
/// coordinates (a Newton solve in type_real) times the face size. A swapped
/// axis or a wrong pair moves the reconstructed point by a sizeable fraction
/// of the face size, many orders of magnitude above this.
constexpr double coordinate_rel_tol =
    std::is_same_v<type_real, float> ? 1e-4 : 1e-9;

/// Collects failures of one kind: the total count and the first few messages.
struct Failures {
  static constexpr int max_listed = 5;
  int count = 0;
  std::vector<std::string> first;

  void add(const std::string &message) {
    if (count < max_listed) {
      first.push_back(message);
    }
    ++count;
  }

  std::string summary(const std::string &title) const {
    std::ostringstream oss;
    oss << title << ": " << count << " failure(s)\n";
    for (const auto &message : first) {
      oss << "    " << message << "\n";
    }
    return oss.str();
  }
};

/// Result of checking all containers of one assembly.
struct Report {
  int containers_checked = 0;
  int pairs_checked = 0;
  long nodes_checked = 0;
  double max_partition_error = 0;
  double max_coordinate_error_rel = 0;
  Failures layout;
  Failures exactly_once;
  Failures partition_of_unity;
  Failures coordinates;
};

/// Everything the host selection test needs to know about one container.
struct ContainerInfo {
  specfem::element_coupling::interface_tag interface_tag;
  specfem::element::boundary_tag boundary_tag;
  int num_pairs;
  bool self_is_host;
  std::size_t extents[5];
};

inline std::string
container_label(const specfem::element_coupling::interface_tag &interface_tag,
                const specfem::element::boundary_tag &boundary_tag) {
  std::ostringstream oss;
  oss << "interface = " << interface_tag
      << ", boundary = " << specfem::element::to_string(boundary_tag);
  return oss.str();
}

template <typename MeshType>
std::unique_ptr<assembly_type> build_assembly(const MeshType &mesh,
                                              const meshing_type &type) {
  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  std::vector<std::shared_ptr<
      specfem::sources::source<specfem::element::dimension_tag::dim3>>>
      sources;
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
      receivers;

  specfem::element_coupling::flux_scheme_configuration flux_scheme_config;
  flux_scheme_config.set_interfacial_meshing_type(type);

  return std::make_unique<assembly_type>(
      mesh, quadrature, sources, receivers,
      std::vector<specfem::enums::wavefield>{}, 1.0, 0.0, 1, 1, 1,
      specfem::simulation::type::forward, false, nullptr, flux_scheme_config);
}

inline double distance(const assembly_type &assembly, const int &ispec1,
                       const int &iz1, const int &iy1, const int &ix1,
                       const int &ispec2, const int &iz2, const int &iy2,
                       const int &ix2) {
  double sum = 0;
  for (int d = 0; d < 3; ++d) {
    const double diff = assembly.mesh.h_coord(ispec1, iz1, iy1, ix1, d) -
                        assembly.mesh.h_coord(ispec2, iz2, iy2, ix2, d);
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

/**
 * @brief Checks the stored interpolants of one container whose self side is
 * the host: extents, exactly-once matching, partition of unity and coordinate
 * reproduction. Appends to `report`.
 */
template <typename ElementTags>
void check_host_container(const assembly_type &assembly,
                          const bool require_host, Report &report) {
  constexpr auto interface_tag = ElementTags::interface_tag;
  constexpr auto boundary_tag = ElementTags::boundary_tag;
  constexpr auto connection_tag = ElementTags::connection_tag;
  constexpr auto flux_scheme_tag = ElementTags::flux_scheme_tag;

  const auto &container =
      assembly.nonconforming_interfaces.template get_interface_container<
          interface_tag, boundary_tag, connection_tag, flux_scheme_tag>();
  const auto [self_faces, coupled_faces] =
      assembly.element_intersections.get_intersections_on_host(
          connection_tag, interface_tag, boundary_tag, flux_scheme_tag);
  const int num_pairs = self_faces.N;
  if (num_pairs == 0) {
    return;
  }

  const std::string label = container_label(interface_tag, boundary_tag);
  const int ngll = assembly.mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::ngllz;

  if (!container.self_is_host) {
    if (require_host) {
      report.layout.add(label + ": non-empty container is not a host");
    }
    return;
  }
  const auto &h = container.h_interpolants;
  if (h.extent(0) != static_cast<std::size_t>(num_pairs) ||
      h.extent(1) != static_cast<std::size_t>(ngll) ||
      h.extent(2) != static_cast<std::size_t>(ngll) ||
      h.extent(3) != static_cast<std::size_t>(ngll) || h.extent(4) != 2) {
    std::ostringstream oss;
    oss << label << ": h_interpolants extents (" << h.extent(0) << ", "
        << h.extent(1) << ", " << h.extent(2) << ", " << h.extent(3) << ", "
        << h.extent(4) << ") != (" << num_pairs << ", " << ngll << ", " << ngll
        << ", " << ngll << ", 2)";
    report.layout.add(oss.str());
    return;
  }

  ++report.containers_checked;
  report.pairs_checked += num_pairs;

  // pairs sharing a self face: key = (self element, self face type)
  std::map<std::pair<int, int>, std::vector<int>> groups;
  for (int ipair = 0; ipair < num_pairs; ++ipair) {
    const auto face = self_faces(ipair);
    groups[{ face.element_index, static_cast<int>(face.face_type) }].push_back(
        ipair);
  }

  for (const auto &[key, pairs] : groups) {
    const auto face_type =
        static_cast<specfem::mesh_entity::dim3::type>(key.second);
    for (int i = 0; i < ngll; ++i) {
      for (int j = 0; j < ngll; ++j) {
        ++report.nodes_checked;

        // Rows that were never written are exactly 0: the view is
        // zero-initialised by Kokkos and only the closest pair stores a row.
        // Hence the exact comparison here, and a stored row always has a
        // nonzero entry (its entries sum to 1).
        int num_nonzero_rows = 0;
        int matched = -1;
        for (const int ipair : pairs) {
          bool any_nonzero = false;
          for (int k = 0; k < ngll; ++k) {
            for (int axis = 0; axis < 2; ++axis) {
              if (h(ipair, i, j, k, axis) != static_cast<type_real>(0)) {
                any_nonzero = true;
              }
            }
          }
          if (any_nonzero) {
            ++num_nonzero_rows;
            matched = ipair;
          }
        }
        if (num_nonzero_rows != 1) {
          std::ostringstream oss;
          oss << label << ": element " << key.first << ", face "
              << specfem::mesh_entity::dim3::to_string(face_type) << ", node ("
              << i << ", " << j << "): " << num_nonzero_rows << " of "
              << pairs.size() << " pairs have a nonzero row, "
              << "expected exactly 1";
          report.exactly_once.add(oss.str());
          continue;
        }

        // partition of unity
        for (int axis = 0; axis < 2; ++axis) {
          double sum = 0;
          for (int k = 0; k < ngll; ++k) {
            sum += h(matched, i, j, k, axis);
          }
          report.max_partition_error =
              std::max(report.max_partition_error, std::abs(sum - 1.0));
          if (std::abs(sum - 1.0) > partition_of_unity_tol) {
            std::ostringstream oss;
            oss << label << ": element " << key.first << ", face "
                << specfem::mesh_entity::dim3::to_string(face_type)
                << ", node (" << i << ", " << j << "), axis " << axis
                << ": sum = " << sum << " (|sum - 1| > "
                << partition_of_unity_tol << ")";
            report.partition_of_unity.add(oss.str());
          }
        }

        // coordinate reproduction: sum_{ki,kj} l_ki(xi) l_kj(eta) x(ki,kj)
        const auto host_point = self_faces(matched, i, j);
        const auto c00 = coupled_faces(matched, 0, 0);
        const auto c11 = coupled_faces(matched, ngll - 1, ngll - 1);
        const auto c01 = coupled_faces(matched, 0, ngll - 1);
        const auto c10 = coupled_faces(matched, ngll - 1, 0);
        const double face_size =
            std::max(distance(assembly, c00.ispec, c00.iz, c00.iy, c00.ix,
                              c11.ispec, c11.iz, c11.iy, c11.ix),
                     distance(assembly, c01.ispec, c01.iz, c01.iy, c01.ix,
                              c10.ispec, c10.iz, c10.iy, c10.ix));

        double error2 = 0;
        double reconstructed[3];
        for (int d = 0; d < 3; ++d) {
          double value = 0;
          for (int ki = 0; ki < ngll; ++ki) {
            for (int kj = 0; kj < ngll; ++kj) {
              const auto guest = coupled_faces(matched, ki, kj);
              value += static_cast<double>(h(matched, i, j, ki, 0)) *
                       static_cast<double>(h(matched, i, j, kj, 1)) *
                       assembly.mesh.h_coord(guest.ispec, guest.iz, guest.iy,
                                             guest.ix, d);
            }
          }
          reconstructed[d] = value;
          const double expected = assembly.mesh.h_coord(
              host_point.ispec, host_point.iz, host_point.iy, host_point.ix, d);
          error2 += (value - expected) * (value - expected);
        }
        const double error = std::sqrt(error2);
        report.max_coordinate_error_rel =
            std::max(report.max_coordinate_error_rel, error / face_size);
        if (error > coordinate_rel_tol * face_size) {
          std::ostringstream oss;
          oss << label << ": element " << key.first << ", face "
              << specfem::mesh_entity::dim3::to_string(face_type) << ", node ("
              << i << ", " << j << "), matched pair " << matched
              << " (guest element " << c00.ispec << "): reconstructed ("
              << reconstructed[0] << ", " << reconstructed[1] << ", "
              << reconstructed[2] << ") vs host node ("
              << assembly.mesh.h_coord(host_point.ispec, host_point.iz,
                                       host_point.iy, host_point.ix, 0)
              << ", "
              << assembly.mesh.h_coord(host_point.ispec, host_point.iz,
                                       host_point.iy, host_point.ix, 1)
              << ", "
              << assembly.mesh.h_coord(host_point.ispec, host_point.iz,
                                       host_point.iy, host_point.ix, 2)
              << "), distance " << error << " > " << coordinate_rel_tol
              << " * face size " << face_size;
          report.coordinates.add(oss.str());
        }
      }
    }
  }
}

/// Runs check_host_container over every nonconforming tag combination. With
/// `require_host`, a non-empty container that is not a host is a failure;
/// otherwise such containers are skipped.
inline Report check_all_hosts(const assembly_type &assembly,
                              const bool require_host) {
  Report report;
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<
          specfem::element::dimension_tag::dim3>{} *
          CONNECTION_SET(nonconforming) *
          INTERFACE_SET(elastic_acoustic, acoustic_elastic) *
          BOUNDARY_SET(none, acoustic_free_surface, stacey,
                       composite_stacey_dirichlet) *
          FLUX_SCHEME_SET(natural),
      [&]<typename ElementTags>() {
        check_host_container<ElementTags>(assembly, require_host, report);
      });
  return report;
}

/// Collects the host flag and interpolant extents of every container.
inline std::vector<ContainerInfo>
collect_container_info(const assembly_type &assembly) {
  std::vector<ContainerInfo> infos;
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<
          specfem::element::dimension_tag::dim3>{} *
          CONNECTION_SET(nonconforming) *
          INTERFACE_SET(elastic_acoustic, acoustic_elastic) *
          BOUNDARY_SET(none, acoustic_free_surface, stacey,
                       composite_stacey_dirichlet) *
          FLUX_SCHEME_SET(natural),
      [&]<typename ElementTags>() {
        const auto &container =
            assembly.nonconforming_interfaces.template get_interface_container<
                ElementTags::interface_tag, ElementTags::boundary_tag,
                ElementTags::connection_tag, ElementTags::flux_scheme_tag>();
        const auto [self_faces, coupled_faces] =
            assembly.element_intersections.get_intersections_on_host(
                ElementTags::connection_tag, ElementTags::interface_tag,
                ElementTags::boundary_tag, ElementTags::flux_scheme_tag);
        ContainerInfo info;
        info.interface_tag = ElementTags::interface_tag;
        info.boundary_tag = ElementTags::boundary_tag;
        info.num_pairs = self_faces.N;
        info.self_is_host = container.self_is_host;
        for (int r = 0; r < 5; ++r) {
          info.extents[r] = container.interpolants.extent(r);
        }
        infos.push_back(info);
      });
  return infos;
}

inline void expect_report_clean(const Report &report,
                                const std::string &database_file) {
  EXPECT_GT(report.containers_checked, 0) << database_file;
  EXPECT_GT(report.nodes_checked, 0) << database_file;
  EXPECT_EQ(report.layout.count, 0)
      << database_file << "\n"
      << report.layout.summary("container layout");
  EXPECT_EQ(report.exactly_once.count, 0)
      << database_file << "\n"
      << report.exactly_once.summary("exactly-once matching");
  EXPECT_EQ(report.partition_of_unity.count, 0)
      << database_file << " (max |sum - 1| = " << report.max_partition_error
      << ")\n"
      << report.partition_of_unity.summary("partition of unity");
  EXPECT_EQ(report.coordinates.count, 0)
      << database_file
      << " (max relative error = " << report.max_coordinate_error_rel << ")\n"
      << report.coordinates.summary("coordinate reproduction");
}

/// Every container is a host: check all of them.
inline void test_self_host_interpolants(const std::string &database_file) {
  const auto mesh =
      specfem::io::read_3d_mesh(database_file, specfem::attenuation::Setup{});
  const auto assembly = build_assembly(mesh, meshing_type::self_host);
  const auto report = check_all_hosts(*assembly, true);
  expect_report_clean(report, database_file);
}

/// Mean over the self faces of one interface tag of the largest
/// corner-to-corner distance of the face (boundary tag none).
inline double mean_self_face_size(
    const assembly_type &assembly,
    const specfem::element_coupling::interface_tag &interface_tag) {
  const auto [self_faces, coupled_faces] =
      assembly.element_intersections.get_intersections_on_host(
          specfem::element_connections::type::nonconforming, interface_tag,
          specfem::element::boundary_tag::none,
          specfem::element_coupling::flux_scheme_tag::natural);
  const int ngll = self_faces.n_points;
  double total = 0;
  for (int ipair = 0; ipair < self_faces.N; ++ipair) {
    const auto c00 = self_faces(ipair, 0, 0);
    const auto c11 = self_faces(ipair, ngll - 1, ngll - 1);
    const auto c01 = self_faces(ipair, 0, ngll - 1);
    const auto c10 = self_faces(ipair, ngll - 1, 0);
    total += std::max(distance(assembly, c00.ispec, c00.iz, c00.iy, c00.ix,
                               c11.ispec, c11.iz, c11.iy, c11.ix),
                      distance(assembly, c01.ispec, c01.iz, c01.iy, c01.ix,
                               c10.ispec, c10.iz, c10.iy, c10.ix));
  }
  return self_faces.N > 0 ? total / self_faces.N : 0.0;
}

/// The finer side hosts. data/dim3/interfaces_only/README.md: N x N acoustic
/// elements lie on M x M elastic elements filling the same square; the
/// acoustic elements are the smaller ones in both fixtures (11 vs 7 and 13 vs
/// 10 elements per side). So acoustic_elastic (self medium: acoustic) must be
/// the host. The geometry of the intersections is checked independently
/// below.
inline void test_highres_host_selection(const std::string &database_file) {
  const auto mesh =
      specfem::io::read_3d_mesh(database_file, specfem::attenuation::Setup{});
  const auto assembly = build_assembly(mesh, meshing_type::highres_host);
  const auto infos = collect_container_info(*assembly);
  const int ngll = assembly->mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::ngllz;

  using specfem::element_coupling::interface_tag;
  constexpr auto finer_interface_tag = interface_tag::acoustic_elastic;
  EXPECT_LT(mean_self_face_size(*assembly, finer_interface_tag),
            mean_self_face_size(*assembly, interface_tag::elastic_acoustic))
      << database_file << ": README says the acoustic side is finer";
  int num_nonempty = 0;
  int num_acoustic_elastic_pairs = 0;
  int num_elastic_acoustic_pairs = 0;
  for (const auto &info : infos) {
    const std::string label =
        container_label(info.interface_tag, info.boundary_tag);
    // HostSelection decides per interface tag, so an empty container has the
    // same flag as every other container of its tag.
    const bool expect_host = info.interface_tag == finer_interface_tag;
    EXPECT_EQ(info.self_is_host, expect_host)
        << database_file << ": " << label << ", pairs = " << info.num_pairs;
    if (info.num_pairs == 0) {
      EXPECT_EQ(info.extents[0], 0u) << database_file << ": " << label;
      continue;
    }
    ++num_nonempty;
    (info.interface_tag == interface_tag::acoustic_elastic
         ? num_acoustic_elastic_pairs
         : num_elastic_acoustic_pairs) += info.num_pairs;

    if (info.self_is_host) {
      EXPECT_EQ(info.extents[0], static_cast<std::size_t>(info.num_pairs))
          << database_file << ": " << label;
      for (int r = 1; r < 4; ++r) {
        EXPECT_EQ(info.extents[r], static_cast<std::size_t>(ngll))
            << database_file << ": " << label << ", extent " << r;
      }
      EXPECT_EQ(info.extents[4], 2u) << database_file << ": " << label;
    } else {
      EXPECT_EQ(info.extents[0], 0u)
          << database_file << ": " << label
          << ": a non-host container must not allocate interpolants";
    }
  }
  EXPECT_GT(num_nonempty, 0) << database_file;
  // Each physical pair is filed under both interface tags.
  EXPECT_GT(num_acoustic_elastic_pairs, 0) << database_file;
  EXPECT_GT(num_elastic_acoustic_pairs, 0) << database_file;

  // Exactly one of an interface tag and its conjugate is the host, for any
  // two containers of different interface tags (whatever their boundary tags,
  // empty or not).
  for (const auto &a : infos) {
    for (const auto &b : infos) {
      if (a.interface_tag == b.interface_tag) {
        continue;
      }
      EXPECT_NE(a.self_is_host, b.self_is_host)
          << database_file << ": "
          << container_label(a.interface_tag, a.boundary_tag) << " and "
          << container_label(b.interface_tag, b.boundary_tag)
          << " must have exactly one host";
    }
  }

  // The hosts that were selected store correct interpolants too.
  const auto report = check_all_hosts(*assembly, false);
  expect_report_clean(report, database_file);
}

} // namespace specfem::nonconforming_test::interpolants_impl

TEST(NonconformingContainer3D, interpolants_self_host_11_7) {
  specfem::nonconforming_test::interpolants_impl::test_self_host_interpolants(
      "data/dim3/interfaces_only/acoustic_elastic_11-7/database.bin");
}

TEST(NonconformingContainer3D, interpolants_self_host_13_10) {
  specfem::nonconforming_test::interpolants_impl::test_self_host_interpolants(
      "data/dim3/interfaces_only/acoustic_elastic_13-10/database.bin");
}

TEST(NonconformingContainer3D, host_selection_highres_host_11_7) {
  specfem::nonconforming_test::interpolants_impl::test_highres_host_selection(
      "data/dim3/interfaces_only/acoustic_elastic_11-7/database.bin");
}

TEST(NonconformingContainer3D, host_selection_highres_host_13_10) {
  specfem::nonconforming_test::interpolants_impl::test_highres_host_selection(
      "data/dim3/interfaces_only/acoustic_elastic_13-10/database.bin");
}
