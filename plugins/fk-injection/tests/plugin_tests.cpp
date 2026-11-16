// Tests of the FK plugin's setup and fill through synthetic SDK structs.

#include "field_helpers.hpp"
#include "fk_injection/fk_injection.hpp"
#include "fk_injection_plugin.hpp"
#include "plugin_config.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fk_injection_test {

// 30 km of water over a half-space.
constexpr const char *fluid_over_solid = R"yaml(
layers:
  - acoustic: { rho: 1000, vp: 1500, thickness: 30000 }
  - elastic:  { rho: 2700, vp: 3000, vs: 1732.051, thickness: 0 }
incidence:
  type: P
  back-azimuth: 30
  take-off: 20
  origin: [0, 0, -224000]
time-window: { frequency-max: 0.2, frequency-sampling: 2, length: 180 }
)yaml";

// Two solid layers, with z = -30 km the interface.
constexpr const char *two_elastic_layers = R"yaml(
layers:
  - elastic: { rho: 2500, vp: 2500, vs: 1250, thickness: 30000 }
  - elastic: { rho: 2700, vp: 3000, vs: 1732.051, thickness: 0 }
incidence:
  type: P
  back-azimuth: 0
  take-off: 20
  origin: [0, 0, -224000]
time-window: { frequency-max: 0.2, frequency-sampling: 2, length: 180 }
)yaml";

constexpr int nstep = 600;
constexpr double dt = 0.15;
// Horizontal half-width of the mesh box; holds every test point.
constexpr double box_half_width = 10000.0;
constexpr double box_depth = 100000.0;

specfem::plugin::SimulationInfo
make_info(const double t0 = 0.0, const double zmax = 0.0,
          const double half_width = box_half_width) {
  specfem_simulation_info info{};
  info.struct_size = sizeof(specfem_simulation_info);
  info.dimension = 3;
  info.nstep = nstep;
  info.t0 = t0;
  info.dt = dt;
  info.bounds[0] = info.bounds[2] = -half_width;
  info.bounds[1] = info.bounds[3] = half_width;
  info.bounds[4] = zmax - box_depth;
  info.bounds[5] = zmax;
  return specfem::plugin::SimulationInfo(info);
}

struct TestPoint {
  double x, y, z;
  double rho, vp, vs;
  double nx = 0, ny = 0, nz = -1;
};

// Layout of a group's buffer, to prove fill goes through the slots rather than
// assuming the sample layout.
constexpr specfem_quantity_slot elastic_slots[] = {
  { specfem_quantity_traction, 0, 3, 0 }, { specfem_quantity_velocity, 3, 3, 0 }
};
constexpr specfem_quantity_slot acoustic_slots[] = {
  { specfem_quantity_potential_rate, 0, 1, 0 },
  { specfem_quantity_displacement, 1, 3, 0 }
};
constexpr const char *elastic_property_names[] = { "kappa", "mu", "rho" };
constexpr const char *acoustic_property_names[] = { "rho_inverse", "kappa" };

/// ABI structs of acoustic points followed by elastic points.
class SyntheticPoints {
public:
  SyntheticPoints(const std::vector<TestPoint> &acoustic,
                  const std::vector<TestPoint> &elastic) {
    add_group(specfem_medium_acoustic, acoustic, acoustic_property_names, 2,
              acoustic_slots, 2, 4,
              [](const TestPoint &point, std::vector<double> &properties) {
                properties.push_back(1.0 / point.rho);
                properties.push_back(point.rho * point.vp * point.vp);
              });
    add_group(specfem_medium_elastic, elastic, elastic_property_names, 3,
              elastic_slots, 2, 6,
              [](const TestPoint &point, std::vector<double> &properties) {
                const double mu = point.rho * point.vs * point.vs;
                properties.push_back(point.rho * point.vp * point.vp -
                                     4.0 * mu / 3.0);
                properties.push_back(mu);
                properties.push_back(point.rho);
              });
    points_.struct_size = sizeof(specfem_points);
    points_.ngroups = static_cast<std::int32_t>(group_pointers_.size());
    points_.n = static_cast<std::int64_t>(x_.size());
    points_.x = x_.data();
    points_.y = y_.data();
    points_.z = z_.data();
    points_.nx = nx_.data();
    points_.ny = ny_.data();
    points_.nz = nz_.data();
    points_.groups = group_pointers_.data();
  }

  specfem::plugin::Points points() const {
    return specfem::plugin::Points(points_);
  }

  const std::vector<std::int64_t> &group_sizes() const { return sizes_; }

  /// Rename a property of a group, so that the plugin can't find it.
  void set_property_name(const int group, const int index, const char *name) {
    names_[group][index] = name;
  }

private:
  template <typename Properties>
  void add_group(const specfem_medium medium,
                 const std::vector<TestPoint> &group_points,
                 const char *const *names, const int nproperties,
                 const specfem_quantity_slot *slots, const int nslots,
                 const int ncomponents, Properties write_properties) {
    auto &properties = properties_.emplace_back();
    auto &group_names = names_.emplace_back(names, names + nproperties);
    specfem_point_group group{};
    group.struct_size = sizeof(specfem_point_group);
    group.medium = medium;
    group.property = specfem_property_isotropic;
    group.nproperties = nproperties;
    group.begin = static_cast<std::int64_t>(x_.size());
    group.count = static_cast<std::int64_t>(group_points.size());
    for (const auto &point : group_points) {
      x_.push_back(point.x);
      y_.push_back(point.y);
      z_.push_back(point.z);
      nx_.push_back(point.nx);
      ny_.push_back(point.ny);
      nz_.push_back(point.nz);
      write_properties(point, properties);
    }
    group.property_names = group_names.data();
    group.properties = properties.data();
    group.ncomponents = ncomponents;
    group.nslots = nslots;
    group.slots = slots;
    sizes_.push_back(group.count);
    group_pointers_.push_back(&groups_.emplace_back(group));
  }

  std::vector<double> x_, y_, z_, nx_, ny_, nz_;
  std::list<std::vector<double>> properties_;
  std::vector<std::vector<const char *>> names_ =
      std::vector<std::vector<const char *>>();
  std::list<specfem_point_group> groups_;
  std::vector<const specfem_point_group *> group_pointers_;
  std::vector<std::int64_t> sizes_;
  specfem_points points_{};
};

/// One group's values, row-major (point, component, step).
using GroupValues = std::vector<double>;

constexpr int acoustic_components = 4;
constexpr int elastic_components = 6;

// Fills through buffers whose strides leave gaps and put steps slowest.
template <typename T>
std::vector<GroupValues> fill_window(fk_injection::FkInjectionPlugin &plugin,
                                     const SyntheticPoints &points,
                                     const int first_step, const int nsteps) {
  constexpr std::int64_t gap = 2;
  const int components[] = { acoustic_components, elastic_components };
  const auto &sizes = points.group_sizes();

  std::vector<std::vector<T>> storage(2);
  std::vector<specfem::plugin::Buffer> buffers;
  for (int g = 0; g < 2; ++g) {
    storage[g].assign(gap * components[g] * sizes[g] * nsteps, T(-999));
    specfem_buffer buffer{};
    buffer.struct_size = sizeof(specfem_buffer);
    buffer.dtype = std::is_same_v<T, float> ? specfem_dtype_float32
                                            : specfem_dtype_float64;
    buffer.data = storage[g].data();
    buffer.npoints = sizes[g];
    buffer.ncomponents = components[g];
    buffer.nsteps = nsteps;
    buffer.stride_component = gap;
    buffer.stride_point = gap * components[g];
    buffer.stride_step = gap * components[g] * sizes[g];
    buffers.emplace_back(buffer);
  }
  plugin.fill(first_step, buffers);

  std::vector<GroupValues> values(2);
  for (int g = 0; g < 2; ++g) {
    for (std::int64_t p = 0; p < sizes[g]; ++p)
      for (int c = 0; c < components[g]; ++c)
        for (int s = 0; s < nsteps; ++s)
          values[g].push_back(static_cast<double>(
              storage[g][p * gap * components[g] + c * gap +
                         s * gap * components[g] * sizes[g]]));
  }
  return values;
}

double peak(const GroupValues &values) {
  double result = 0.0;
  for (const double value : values)
    result = std::max(result, std::fabs(value));
  return result;
}

void expect_close(const GroupValues &actual, const GroupValues &expected,
                  const double relative_tolerance) {
  ASSERT_EQ(actual.size(), expected.size());
  const double tolerance = relative_tolerance * peak(expected);
  for (std::size_t i = 0; i < actual.size(); ++i)
    ASSERT_NEAR(actual[i], expected[i], tolerance) << "value " << i;
}

std::vector<TestPoint> acoustic_points() {
  return { { 1000, -2000, -10000, 1000, 1500, 0, 1, 0, 0 },
           { -3000, 4000, -20000, 1000, 1500, 0, 0, 1, 0 } };
}

std::vector<TestPoint> elastic_points() {
  return { { 5000, 1000, -40000, 2700, 3000, 1732.051, 1, 0, 0 },
           { -2000, -6000, -40000, 2700, 3000, 1732.051, 0, -1, 0 },
           { 0, 3000, -60000, 2700, 3000, 1732.051, 0, 0, -1 } };
}

// The field at the points from the response, table and sampler, in the buffer
// layout of SyntheticPoints.
std::vector<GroupValues> direct_field(const std::string &config_text,
                                      const std::vector<TestPoint> &acoustic,
                                      const std::vector<TestPoint> &elastic,
                                      const int first_step, const int nsteps) {
  const auto config = fk_injection::parse_plugin_config(config_text);
  std::vector<fk_injection::FieldPoint> points;
  const auto add = [&](const TestPoint &point,
                       const specfem::plugin::Medium medium) {
    fk_injection::FieldPoint &field_point = points.emplace_back();
    field_point.x = static_cast<fk_injection::type_real>(point.x);
    field_point.y = static_cast<fk_injection::type_real>(point.y);
    field_point.depth = static_cast<fk_injection::type_real>(-point.z);
    field_point.medium = medium;
    field_point.normal_x = static_cast<fk_injection::type_real>(point.nx);
    field_point.normal_y = static_cast<fk_injection::type_real>(point.ny);
    field_point.normal_z = static_cast<fk_injection::type_real>(point.nz);
    if (medium == specfem::plugin::Medium::elastic) {
      const double xi = point.vs * point.vs / (point.vp * point.vp);
      const double mu = point.rho * point.vs * point.vs;
      field_point.xi1 = static_cast<fk_injection::type_real>(1.0 - 2.0 * xi);
      field_point.xim = static_cast<fk_injection::type_real>((1.0 - xi) * mu);
      // bulk = lambda / (2 (lambda + mu)) is Poisson's ratio.
      field_point.bulk = static_cast<fk_injection::type_real>(
          (1.0 - 2.0 * xi) / (2.0 * (1.0 - xi)));
    }
  };
  for (const auto &point : acoustic)
    add(point, specfem::plugin::Medium::acoustic);
  for (const auto &point : elastic)
    add(point, specfem::plugin::Medium::elastic);

  const auto result = fk_injection_tests_impl::field_at(
      config.model, config.incident_wave(0.0, 0.0),
      config.time_window(static_cast<fk_injection::type_real>(dt), nstep),
      { -box_half_width, box_half_width, -box_half_width, box_half_width },
      points);
  std::vector<fk_injection::type_real> samples(
      points.size() * fk_injection::number_of_sample_components * nsteps);
  result.sample(first_step, nsteps, samples);
  const auto at = [&](const std::size_t point, const int component,
                      const int step) {
    return static_cast<double>(
        samples[(point * fk_injection::number_of_sample_components +
                 component) *
                    nsteps +
                step]);
  };

  std::vector<GroupValues> values(2);
  for (std::size_t p = 0; p < acoustic.size(); ++p) {
    for (int s = 0; s < nsteps; ++s)
      values[0].push_back(at(p, fk_injection::sample_potential_rate_offset, s));
    for (int c = 0; c < 3; ++c)
      for (int s = 0; s < nsteps; ++s)
        values[0].push_back(
            at(p, fk_injection::sample_kinematic_offset + c, s));
  }
  for (std::size_t p = 0; p < elastic.size(); ++p) {
    const std::size_t point = acoustic.size() + p;
    for (int c = 0; c < 3; ++c)
      for (int s = 0; s < nsteps; ++s)
        values[1].push_back(
            at(point, fk_injection::sample_traction_offset + c, s));
    for (int c = 0; c < 3; ++c)
      for (int s = 0; s < nsteps; ++s)
        values[1].push_back(
            at(point, fk_injection::sample_kinematic_offset + c, s));
  }
  return values;
}

fk_injection::FkInjectionPlugin
make_plugin(const std::string &config_text, const double t0 = 0.0,
            const double zmax = 0.0, const double half_width = box_half_width) {
  return fk_injection::FkInjectionPlugin(
      fk_injection::parse_plugin_config(config_text),
      make_info(t0, zmax, half_width));
}

template <typename T> class FkPluginBuffers : public ::testing::Test {};
using BufferTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(FkPluginBuffers, BufferTypes);

// Every slot reads what the sampler gives at that point, with a Lame ratio
// from the textbook formula.
TYPED_TEST(FkPluginBuffers, MatchDirectFieldPerSlot) {
  const SyntheticPoints points(acoustic_points(), elastic_points());
  auto plugin = make_plugin(fluid_over_solid);
  plugin.setup(points.points());

  const auto actual = fill_window<TypeParam>(plugin, points, 0, nstep);
  const auto expected = direct_field(fluid_over_solid, acoustic_points(),
                                     elastic_points(), 0, nstep);
  for (int g = 0; g < 2; ++g) {
    ASSERT_GT(peak(expected[g]), 0.0);
    expect_close(actual[g], expected[g], 1e-5);
  }
}

} // namespace fk_injection_test

TEST(FkPlugin, TypeRealFollowsPrecisionOption) {
#ifdef FK_INJECTION_EXPECT_DOUBLE
  EXPECT_TRUE((std::is_same_v<fk_injection::type_real, double>));
#else
  EXPECT_TRUE((std::is_same_v<fk_injection::type_real, float>));
#endif
}

TEST(FkPlugin, SupportsExactlyWhatInjectionNeeds) {
  auto plugin =
      fk_injection_test::make_plugin(fk_injection_test::fluid_over_solid);
  using specfem::plugin::Medium;
  using specfem::plugin::Property;
  using specfem::plugin::Quantity;
  const std::vector<Quantity> quantities = {
    Quantity::displacement, Quantity::velocity, Quantity::acceleration,
    Quantity::traction, Quantity::potential_rate
  };
  for (const Quantity quantity : quantities) {
    EXPECT_EQ(plugin.supports(Medium::elastic, Property::isotropic, quantity),
              quantity == Quantity::velocity || quantity == Quantity::traction);
    EXPECT_EQ(plugin.supports(Medium::acoustic, Property::isotropic, quantity),
              quantity == Quantity::displacement ||
                  quantity == Quantity::potential_rate);
    EXPECT_FALSE(
        plugin.supports(Medium::elastic_spin, Property::isotropic, quantity));
    EXPECT_FALSE(
        plugin.supports(Medium::elastic, Property::anisotropic, quantity));
    EXPECT_FALSE(
        plugin.supports(Medium::acoustic, Property::anisotropic, quantity));
  }
}

// ABI: the same values regardless of window order or size.
TEST(FkPlugin, WindowsInAnyOrderMatchOneWindow) {
  const fk_injection_test::SyntheticPoints points(
      fk_injection_test::acoustic_points(),
      fk_injection_test::elastic_points());
  auto plugin =
      fk_injection_test::make_plugin(fk_injection_test::fluid_over_solid);
  plugin.setup(points.points());

  const int nsteps = 40;
  const auto whole =
      fk_injection_test::fill_window<double>(plugin, points, 300, 2 * nsteps);
  const auto second = fk_injection_test::fill_window<double>(
      plugin, points, 300 + nsteps, nsteps);
  const auto first =
      fk_injection_test::fill_window<double>(plugin, points, 300, nsteps);

  const int components[] = { fk_injection_test::acoustic_components,
                             fk_injection_test::elastic_components };
  for (int g = 0; g < 2; ++g) {
    ASSERT_GT(fk_injection_test::peak(whole[g]), 0.0);
    const std::size_t rows = whole[g].size() / (2 * nsteps);
    ASSERT_EQ(rows % components[g], 0u);
    for (std::size_t row = 0; row < rows; ++row)
      for (int s = 0; s < nsteps; ++s) {
        EXPECT_EQ(first[g][row * nsteps + s], whole[g][row * 2 * nsteps + s]);
        EXPECT_EQ(second[g][row * nsteps + s],
                  whole[g][row * 2 * nsteps + nsteps + s]);
      }
  }
}

TEST(FkPlugin, EmptyGroupIsFine) {
  const fk_injection_test::SyntheticPoints points(
      {}, fk_injection_test::elastic_points());
  auto plugin =
      fk_injection_test::make_plugin(fk_injection_test::fluid_over_solid);
  plugin.setup(points.points());
  const auto values =
      fk_injection_test::fill_window<double>(plugin, points, 0, 200);
  EXPECT_TRUE(values[0].empty());
  EXPECT_GT(fk_injection_test::peak(values[1]), 0.0);
}

TEST(FkPlugin, RankWithoutPointsFillsNothing) {
  const fk_injection_test::SyntheticPoints points({}, {});
  auto plugin =
      fk_injection_test::make_plugin(fk_injection_test::fluid_over_solid);
  plugin.setup(points.points());
  const auto values =
      fk_injection_test::fill_window<double>(plugin, points, 0, 5);
  EXPECT_TRUE(values[0].empty());
  EXPECT_TRUE(values[1].empty());
}

TEST(FkPlugin, MissingPropertyIsNamed) {
  fk_injection_test::SyntheticPoints points(
      {}, fk_injection_test::elastic_points());
  points.set_property_name(1, 1, "not_mu");
  auto plugin =
      fk_injection_test::make_plugin(fk_injection_test::fluid_over_solid);
  try {
    plugin.setup(points.points());
    FAIL() << "expected a missing property";
  } catch (const std::out_of_range &error) {
    EXPECT_NE(std::string(error.what()).find("mu"), std::string::npos)
        << error.what();
  }
}

// The ray parameter must be valid in every layer; here the fast top layer makes
// the wave evanescent.
TEST(FkPlugin, EvanescentWaveIsUnsupported) {
  auto plugin = fk_injection_test::make_plugin(R"yaml(
layers:
  - elastic: { rho: 2500, vp: 6000, vs: 3000, thickness: 30000 }
  - elastic: { rho: 2700, vp: 3000, vs: 1732.051, thickness: 0 }
incidence: { type: P, azimuth: 0, take-off: 60, origin: [0, 0, -224000] }
time-window: { frequency-max: 0.2, frequency-sampling: 2, length: 180 }
)yaml");
  const fk_injection_test::SyntheticPoints points(
      {}, { { 0, 0, -40000, 2700, 3000, 1732.051 } });
  EXPECT_THROW(plugin.setup(points.points()), specfem::plugin::Unsupported);
}

TEST(FkPlugin, MaterialMismatchIsUnsupportedAndNamesThePoint) {
  auto plugin =
      fk_injection_test::make_plugin(fk_injection_test::fluid_over_solid);
  const fk_injection_test::SyntheticPoints points(
      {}, { { 0, 0, -40000, 2700, 3000, 1732.051 },
            { 123, 456, -50000, 2800, 3000, 1732.051 } });
  try {
    plugin.setup(points.points());
    FAIL() << "expected a material mismatch";
  } catch (const specfem::plugin::Unsupported &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("point 1"), std::string::npos) << message;
    EXPECT_NE(message.find("123"), std::string::npos) << message;
    EXPECT_NE(message.find("2800"), std::string::npos) << message;
    EXPECT_NE(message.find("2700"), std::string::npos) << message;
  }
}

TEST(FkPlugin, ElasticPointInTheFluidColumnIsUnsupported) {
  auto plugin =
      fk_injection_test::make_plugin(fk_injection_test::fluid_over_solid);
  const fk_injection_test::SyntheticPoints points(
      {}, { { 0, 0, -10000, 2700, 3000, 1732.051 } });
  EXPECT_THROW(plugin.setup(points.points()), specfem::plugin::Unsupported);
}

// A GLL point on an interface carries its element's material, on either side.
TEST(FkPlugin, InterfacePointMayCarryEitherLayersMaterial) {
  auto plugin =
      fk_injection_test::make_plugin(fk_injection_test::two_elastic_layers);
  const fk_injection_test::SyntheticPoints on_interface(
      {}, { { 0, 0, -30000, 2500, 2500, 1250 },
            { 0, 0, -30000, 2700, 3000, 1732.051 } });
  ASSERT_NO_THROW(plugin.setup(on_interface.points()));

  // Both read the interface's field. The normal is vertical, so the material's
  // Lame ratios don't enter the traction either.
  const int nsteps = 200;
  const auto values =
      fk_injection_test::fill_window<double>(plugin, on_interface, 0, nsteps);
  const std::size_t point_size = fk_injection_test::elastic_components * nsteps;
  ASSERT_EQ(values[1].size(), 2 * point_size);
  const fk_injection_test::GroupValues upper(values[1].begin(),
                                             values[1].begin() + point_size);
  const fk_injection_test::GroupValues lower(values[1].begin() + point_size,
                                             values[1].end());
  EXPECT_GT(fk_injection_test::peak(upper), 0.0);
  EXPECT_EQ(upper, lower);

  const fk_injection_test::SyntheticPoints above_interface(
      {}, { { 0, 0, -29000, 2700, 3000, 1732.051 } });
  EXPECT_THROW(plugin.setup(above_interface.points()),
               specfem::plugin::Unsupported);
}

// With the default origin-time the wave crosses the origin at the first step,
// so t0 changes nothing.
TEST(FkPlugin, DefaultOriginTimeIsIndependentOfT0) {
  const fk_injection_test::SyntheticPoints points(
      fk_injection_test::acoustic_points(),
      fk_injection_test::elastic_points());
  auto at_zero =
      fk_injection_test::make_plugin(fk_injection_test::fluid_over_solid, 0.0);
  auto at_minus_12 = fk_injection_test::make_plugin(
      fk_injection_test::fluid_over_solid, -12.0);
  at_zero.setup(points.points());
  at_minus_12.setup(points.points());
  const auto expected =
      fk_injection_test::fill_window<double>(at_zero, points, 0, 200);
  const auto actual =
      fk_injection_test::fill_window<double>(at_minus_12, points, 0, 200);
  EXPECT_EQ(actual, expected);
}

// Moving the mesh and the origin together moves nothing relative to the model.
TEST(FkPlugin, ShiftingMeshAndOriginTogetherChangesNothing) {
  const double shift = 1024.0;
  auto shifted_acoustic = fk_injection_test::acoustic_points();
  auto shifted_elastic = fk_injection_test::elastic_points();
  for (auto &point : shifted_acoustic)
    point.z += shift;
  for (auto &point : shifted_elastic)
    point.z += shift;

  std::string shifted_config = fk_injection_test::fluid_over_solid;
  const std::string origin = "origin: [0, 0, -224000]";
  shifted_config.replace(shifted_config.find(origin), origin.size(),
                         "origin: [0, 0, -222976]");

  const fk_injection_test::SyntheticPoints reference_points(
      fk_injection_test::acoustic_points(),
      fk_injection_test::elastic_points());
  const fk_injection_test::SyntheticPoints shifted_points(shifted_acoustic,
                                                          shifted_elastic);
  auto reference =
      fk_injection_test::make_plugin(fk_injection_test::fluid_over_solid);
  auto shifted = fk_injection_test::make_plugin(shifted_config, 0.0, shift);
  reference.setup(reference_points.points());
  shifted.setup(shifted_points.points());

  const auto expected = fk_injection_test::fill_window<double>(
      reference, reference_points, 0, 200);
  const auto actual =
      fk_injection_test::fill_window<double>(shifted, shifted_points, 0, 200);
  for (int g = 0; g < 2; ++g)
    fk_injection_test::expect_close(actual[g], expected[g], 1e-6);
}

// The FFT length and leading pad follow the delays, and the field changes with
// them: by up to 10% here if each rank sizes from its own points. The short
// window keeps the FFT length from clamping to the same value either way.
TEST(FkPlugin, PartitioningDoesNotChangeTheField) {
  std::vector<fk_injection_test::TestPoint> everywhere;
  for (const double x : { -200000.0, -100000.0, 0.0, 100000.0, 200000.0 })
    for (const double y : { -200000.0, 0.0, 200000.0 })
      everywhere.push_back({ x, y, -40000, 2700, 3000, 1732.051, 1, 0, 0 });
  const std::vector<fk_injection_test::TestPoint> one_rank = {
    { 0, 0, -40000, 2700, 3000, 1732.051, 1, 0, 0 },
    { 100000, 0, -40000, 2700, 3000, 1732.051, 1, 0, 0 }
  };

  std::string config = fk_injection_test::fluid_over_solid;
  const std::string length = "length: 180";
  config.replace(config.find(length), length.size(), "length: 100");

  const fk_injection_test::SyntheticPoints all_points({}, everywhere);
  const fk_injection_test::SyntheticPoints rank_points({}, one_rank);
  auto all_plugin = fk_injection_test::make_plugin(config, 0.0, 0.0, 200000.0);
  auto rank_plugin = fk_injection_test::make_plugin(config, 0.0, 0.0, 200000.0);
  all_plugin.setup(all_points.points());
  rank_plugin.setup(rank_points.points());

  const int nsteps = 600;
  const auto all_values =
      fk_injection_test::fill_window<double>(all_plugin, all_points, 0, nsteps);
  const auto rank_values = fk_injection_test::fill_window<double>(
      rank_plugin, rank_points, 0, nsteps);

  // Row-major (point, component, step); index 7 is (0, 0), index 10 (100 km,
  // 0).
  const int indices[] = { 7, 10 };
  const int rows = fk_injection_test::elastic_components * nsteps;
  fk_injection_test::GroupValues subset;
  for (const int index : indices)
    subset.insert(subset.end(), all_values[1].begin() + index * rows,
                  all_values[1].begin() + (index + 1) * rows);
  double difference = 0.0;
  for (std::size_t i = 0; i < subset.size(); ++i)
    difference = std::max(difference, std::fabs(subset[i] - rank_values[1][i]));
  const double relative = difference / fk_injection_test::peak(subset);
  EXPECT_LT(relative, 1e-5) << "relative difference " << relative;
}
