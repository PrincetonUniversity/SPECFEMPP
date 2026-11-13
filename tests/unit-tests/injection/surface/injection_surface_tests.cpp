// Tests for specfem::assembly::InjectionSurface<dim3> and
// specfem::injection::make_injection_points, built from databases whose mesher
// declared injection faces.

#include "../../assembly/dim3/test_fixture.hpp"
#include "../session/mock_plugin.hpp"
#include "specfem/assembly/boundaries/dim3/impl/utilities.hpp"
#include "specfem/injection.hpp"
#include "specfem/injection/points/points.hpp"
#include "specfem/injection/session.hpp"
#include <Kokkos_Core.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specfem::injection_surface_test {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using Surface =
    specfem::assembly::InjectionSurface<specfem::element::dimension_tag::dim3>;

constexpr type_real tolerance = 1e-4;
// Coordinates are O(1e4) m in single precision, so geometric comparisons are
// relative.
constexpr double relative_tolerance = 1e-5;
constexpr int ngll = 5;

// A box side is keyed by 2 * axis + (normal points to +axis).
struct Case {
  std::string folder;  ///< Directory under data/dim3
  std::set<int> sides; ///< Box sides declared as injection sides
};

std::ostream &operator<<(std::ostream &stream, const Case &test_case) {
  return stream << test_case.folder;
}

class InjectionSurface3DTest : public ::testing::TestWithParam<Case> {
protected:
  void SetUp() override {
    fixture = specfem::test_configuration::Assembly3D(
        "data/dim3/" + GetParam().folder + "/database.bin");
    fixture.assembly.t0 = -1.5;
    fixture.assembly.dt = 0.25;
  }

  const auto &assembly() const { return fixture.assembly; }
  const Surface &surface() const { return fixture.assembly.injection_surface; }

  specfem::test_configuration::Assembly3D fixture;
};

int count_systems() {
  int count = 0;
  specfem::tag_dispatch::for_each(Surface::combinations,
                                  [&]<typename TagsType>() { ++count; });
  return count;
}

// Requests velocity and traction on elastic/isotropic points only.
std::optional<std::vector<specfem::injection::QuantityRequest>>
elastic_isotropic_only(const specfem::element::medium_tag medium,
                       const specfem::element::property_tag property) {
  if (medium == specfem::element::medium_tag::elastic &&
      property == specfem::element::property_tag::isotropic)
    return std::vector<specfem::injection::QuantityRequest>{
      { specfem_quantity_velocity, 3 }, { specfem_quantity_traction, 3 }
    };
  return std::nullopt;
}

// The bounding box of the fixtures is finite and non-degenerate; the geometric
// checks below rely on it.
void expect_valid_bounds(
    const specfem::test_configuration::Assembly3D &fixture) {
  const auto &bounds = fixture.assembly.info.domain_bounds.bounds_array;
  for (int d = 0; d < 3; ++d) {
    ASSERT_TRUE(std::isfinite(bounds(d).min)) << "axis " << d;
    ASSERT_TRUE(std::isfinite(bounds(d).max)) << "axis " << d;
    ASSERT_LT(bounds(d).min, bounds(d).max) << "axis " << d;
  }
}

TEST_P(InjectionSurface3DTest, DomainBoundsAreValid) {
  expect_valid_bounds(fixture);
}

// Box side of a normal: the axis of its largest component, and its sign.
int side_of(const std::array<double, 3> &normal) {
  int axis = 0;
  for (int d = 1; d < 3; ++d)
    if (std::abs(normal[d]) > std::abs(normal[axis]))
      axis = d;
  return 2 * axis + (normal[axis] > 0);
}

// The entries are exactly the Stacey points whose stored face lies on an
// injection side: one per (element, GLL point), with Stacey's face.
TEST_P(InjectionSurface3DTest, EntriesAreStaceyPointsOnInjectionSides) {
  const auto &boundaries = assembly().boundaries;
  using Point = std::array<int, 4>;
  std::set<Point> expected;
  for (int ispec = 0; ispec < assembly().mesh.nspec; ++ispec) {
    const int local = boundaries.h_stacey_index_mapping(ispec);
    if (local < 0)
      continue;
    for (int iz = 0; iz < ngll; ++iz)
      for (int iy = 0; iy < ngll; ++iy)
        for (int ix = 0; ix < ngll; ++ix) {
          if (boundaries.stacey.h_face_weight(local, iz, iy, ix) == 0)
            continue;
          std::array<double, 3> normal{};
          for (int d = 0; d < 3; ++d)
            normal[d] = boundaries.stacey.h_face_normal(local, iz, iy, ix, d);
          if (GetParam().sides.count(side_of(normal)))
            expected.insert({ ispec, iz, iy, ix });
        }
  }

  std::set<Point> actual;
  for (int e = 0; e < surface().nentries; ++e)
    actual.insert({ surface().h_ispec(e), surface().h_iz(e), surface().h_iy(e),
                    surface().h_ix(e) });
  EXPECT_EQ(static_cast<int>(actual.size()), surface().nentries)
      << "an (element, GLL point) has more than one entry";
  EXPECT_EQ(actual, expected);

  EXPECT_EQ(surface().nrows, surface().nentries);
  for (int entry = 0; entry < surface().nentries; ++entry) {
    EXPECT_EQ(surface().h_row(entry), entry);
    EXPECT_EQ(surface().h_row_entry(entry), entry);
  }
}

TEST_P(InjectionSurface3DTest, CoordinatesAndTagsMatchAssembly) {
  const auto &coord = assembly().mesh.h_coord;
  for (int e = 0; e < surface().nentries; ++e) {
    const int ispec = surface().h_ispec(e);
    const int iz = surface().h_iz(e), iy = surface().h_iy(e),
              ix = surface().h_ix(e);
    EXPECT_NEAR(surface().x(e), coord(ispec, iz, iy, ix, 0), tolerance);
    EXPECT_NEAR(surface().y(e), coord(ispec, iz, iy, ix, 1), tolerance);
    EXPECT_NEAR(surface().z(e), coord(ispec, iz, iy, ix, 2), tolerance);
    EXPECT_EQ(surface().h_boundary_tag(e),
              specfem::element::boundary_tag::stacey);
    EXPECT_EQ(surface().nx(e), surface().h_normal(e, 0));
  }
}

// Every normal is the outward unit normal of a box side, and the point lies on
// that side. Side d, sign s is keyed as 2 * d + (s > 0).
TEST_P(InjectionSurface3DTest, NormalsPointOutOfTheBox) {
  expect_valid_bounds(fixture);
  const auto &bounds = assembly().info.domain_bounds.bounds_array;
  double extent = 0;
  for (int d = 0; d < 3; ++d)
    extent = std::max<double>(extent, bounds(d).max - bounds(d).min);
  for (int e = 0; e < surface().nentries; ++e) {
    int axis = -1;
    for (int d = 0; d < 3; ++d)
      if (std::abs(std::abs(surface().h_normal(e, d)) - 1) < tolerance)
        axis = d;
    ASSERT_GE(axis, 0) << "entry " << e << " has no axis-aligned normal";
    for (int d = 0; d < 3; ++d)
      if (d != axis)
        EXPECT_NEAR(surface().h_normal(e, d), 0, tolerance);
    const std::array<type_real, 3> point = { surface().x(e), surface().y(e),
                                             surface().z(e) };
    const auto plane =
        surface().h_normal(e, axis) > 0 ? bounds(axis).max : bounds(axis).min;
    EXPECT_NEAR(point[axis], plane, relative_tolerance * extent)
        << "entry " << e;
  }
}

// At every entry, normal * dS equals Stacey's area-scaled normal times its
// GLL weight, so the absorbing and injected terms use the same face.
TEST_P(InjectionSurface3DTest, MatchesStaceyGeometry) {
  const auto &boundaries = assembly().boundaries;
  int compared = 0;
  for (int e = 0; e < surface().nentries; ++e) {
    const int iz = surface().h_iz(e), iy = surface().h_iy(e),
              ix = surface().h_ix(e);
    const int istacey = boundaries.h_stacey_index_mapping(surface().h_ispec(e));
    ASSERT_GE(istacey, 0);
    for (int d = 0; d < 3; ++d) {
      const double expected =
          boundaries.stacey.h_face_normal(istacey, iz, iy, ix, d) *
          boundaries.stacey.h_face_weight(istacey, iz, iy, ix);
      EXPECT_NEAR(surface().h_normal(e, d) * surface().h_weight(e), expected,
                  relative_tolerance * surface().h_weight(e))
          << "entry " << e;
    }
    ++compared;
  }
  EXPECT_EQ(compared, surface().nentries);
}

TEST_P(InjectionSurface3DTest, RangesPartitionEntriesByGroup) {
  int next = 0;
  specfem::tag_dispatch::for_each(
      Surface::combinations, [&]<typename TagsType>() {
        const auto entries = surface().entry_ranges.get<TagsType>();
        EXPECT_EQ(entries.begin_index(), next);
        next = entries.end_index();
        const bool elastic_isotropic =
            TagsType::medium_tag == specfem::element::medium_tag::elastic &&
            TagsType::property_tag == specfem::element::property_tag::isotropic;
        EXPECT_EQ(entries.size(), elastic_isotropic ? surface().nentries : 0);
      });
  EXPECT_EQ(next, surface().nentries);
}

TEST_P(InjectionSurface3DTest, DeviceViewsMatchHost) {
  const auto weight = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          surface().weight);
  const auto row =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), surface().row);
  for (int e = 0; e < surface().nentries; ++e) {
    EXPECT_EQ(weight(e), surface().h_weight(e));
    EXPECT_EQ(row(e), surface().h_row(e));
  }
}

TEST_P(InjectionSurface3DTest, OneGroupPerSystem) {
  const auto points = specfem::injection::make_injection_points(
      assembly(), elastic_isotropic_only, "test");
  ASSERT_EQ(static_cast<int>(points.groups.size()), count_systems());
  EXPECT_EQ(points.size(), surface().nrows);

  // Coordinates and normals are shared with the assembly, not copied.
  EXPECT_EQ(points.x.data(), surface().x.data());
  EXPECT_EQ(points.nz.data(), surface().nz.data());

  std::size_t g = 0;
  specfem::tag_dispatch::for_each(
      Surface::combinations, [&]<typename TagsType>() {
        const auto &group = points.groups[g++];
        const auto rows = surface().row_ranges.get<TagsType>();
        EXPECT_EQ(group.medium,
                  static_cast<specfem_medium>(TagsType::medium_tag));
        EXPECT_EQ(group.property,
                  static_cast<specfem_property>(TagsType::property_tag));
        EXPECT_EQ(group.begin, rows.begin_index());
        EXPECT_EQ(group.count, rows.size());
        EXPECT_FALSE(group.property_names.empty());
        // Only systems with points request quantities.
        EXPECT_EQ(group.quantities.empty(), rows.size() == 0);
      });
}

TEST_P(InjectionSurface3DTest, GroupsCarryNamedProperties) {
  const auto points = specfem::injection::make_injection_points(
      assembly(), elastic_isotropic_only, "test");
  std::size_t g = 0;
  specfem::tag_dispatch::for_each(
      Surface::combinations, [&]<typename TagsType>() {
        const auto &group = points.groups[g++];
        if constexpr (TagsType::medium_tag ==
                          specfem::element::medium_tag::elastic &&
                      TagsType::property_tag ==
                          specfem::element::property_tag::isotropic) {
          EXPECT_THAT(group.property_names, ElementsAre("kappa", "mu", "rho"));
          const auto &container =
              assembly()
                  .properties.template get_container<TagsType::medium_tag,
                                                     TagsType::property_tag>();
          for (int r = 0; r < group.count; ++r) {
            const int e = surface().h_row_entry(group.begin + r);
            const int local =
                assembly()
                    .properties.template property_index_mapping<
                        TagsType::medium_tag, TagsType::property_tag>(
                        surface().h_ispec(e));
            const int iz = surface().h_iz(e), iy = surface().h_iy(e),
                      ix = surface().h_ix(e);
            EXPECT_EQ(group.properties(r, 0),
                      container.h_kappa(local, iz, iy, ix));
            EXPECT_EQ(group.properties(r, 1),
                      container.h_mu(local, iz, iy, ix));
            EXPECT_EQ(group.properties(r, 2),
                      container.h_rho(local, iz, iy, ix));
          }
        }
        if constexpr (TagsType::medium_tag ==
                      specfem::element::medium_tag::acoustic)
          EXPECT_THAT(group.property_names,
                      ElementsAre("rho_inverse", "kappa"));
      });
}

TEST_P(InjectionSurface3DTest, MethodIsAskedOnlyAboutSystemsWithPoints) {
  std::vector<
      std::pair<specfem::element::medium_tag, specfem::element::property_tag>>
      asked;
  const auto recording = [&](const specfem::element::medium_tag medium,
                             const specfem::element::property_tag property) {
    asked.emplace_back(medium, property);
    return elastic_isotropic_only(medium, property);
  };
  [[maybe_unused]] const auto points =
      specfem::injection::make_injection_points(assembly(), recording, "test");
  if (surface().nrows == 0)
    EXPECT_TRUE(asked.empty());
  else
    EXPECT_THAT(asked, ElementsAre(std::pair(
                           specfem::element::medium_tag::elastic,
                           specfem::element::property_tag::isotropic)));
}

TEST_P(InjectionSurface3DTest, UnsupportedSystemWithPointsFails) {
  const auto nothing = [](specfem::element::medium_tag,
                          specfem::element::property_tag) {
    return std::optional<std::vector<specfem::injection::QuantityRequest>>{};
  };
  if (surface().nrows == 0) {
    EXPECT_NO_THROW([[maybe_unused]] const auto points =
                        specfem::injection::make_injection_points(
                            assembly(), nothing, "stacey"));
    return;
  }
  try {
    [[maybe_unused]] const auto points =
        specfem::injection::make_injection_points(assembly(), nothing,
                                                  "stacey");
    FAIL() << "an unsupported system with points should throw";
  } catch (const std::runtime_error &error) {
    EXPECT_THAT(error.what(),
                HasSubstr(std::to_string(surface().nrows) +
                          " points in elastic/isotropic elements, but the "
                          "stacey injection method does not support them"));
  }
}

TEST_P(InjectionSurface3DTest, SimulationParametersUseGlobalBounds) {
  const auto parameters =
      specfem::injection::make_simulation_parameters(assembly(), 12);
  EXPECT_NEAR(parameters.t0, -1.5, tolerance);
  EXPECT_NEAR(parameters.dt, 0.25, tolerance);
  EXPECT_EQ(parameters.nstep, 12);
  const auto &bounds = assembly().info.domain_bounds.bounds_array;
  for (int d = 0; d < 3; ++d) {
    EXPECT_NEAR(parameters.bounds[2 * d], bounds(d).min, tolerance);
    EXPECT_NEAR(parameters.bounds[2 * d + 1], bounds(d).max, tolerance);
  }
}

// The points drive a provider session end to end: the mock plugin receives
// every group and fills values the frame buffer serves by row.
TEST_P(InjectionSurface3DTest, DrivesProviderSession) {
  using specfem::injection_session_test::mock_calls;
  using specfem::injection_session_test::mock_plugin_spec;
  using specfem::injection_session_test::mock_value;
  specfem::injection_session_test::reset_mock_calls();

  specfem::injection::SessionOptions options;
  options.window_steps = 4;
  specfem::injection::ProviderSession session(
      specfem::injection::PluginLibrary::open(mock_plugin_spec()), "ok",
      specfem::injection::make_simulation_parameters(assembly(), 8), options);
  session.setup(specfem::injection::make_injection_points(
      assembly(), elastic_isotropic_only, "test"));
  EXPECT_EQ(static_cast<int>(mock_calls().groups.size()), count_systems());
  EXPECT_EQ(mock_calls().setup_points, surface().nrows);

  session.ensure_window(5);
  if (surface().nrows == 0)
    return;
  const auto &frame = session.frame();
  const int row = surface().nrows - 1;
  Kokkos::View<type_real, Kokkos::DefaultExecutionSpace> value("value");
  Kokkos::parallel_for(
      "read_frame", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(const int) { value() = frame.load(5, row, 4); });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), value);
  EXPECT_NEAR(host(), mock_value(surface().x(row), 4, 5), 1e-3);
}

INSTANTIATE_TEST_SUITE_P(
    InjectionSurface3D, InjectionSurface3DTest,
    ::testing::Values(Case{ "EightNodeElasticInjectionStacey",
                            { 0, 1, 2, 3, 4 } },
                      Case{ "EightNodeElasticInjectionSides", { 0, 4 } },
                      Case{ "EightNodeElastic", {} }),
    [](const ::testing::TestParamInfo<Case> &info) {
      return info.param.folder;
    });

} // namespace specfem::injection_surface_test
