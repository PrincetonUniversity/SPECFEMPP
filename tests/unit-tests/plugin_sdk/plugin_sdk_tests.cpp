// Tests for the injection plugin SDK: the C++ helper types, and real plugin
// modules loaded with dlopen exactly as SPECFEM++ will load them.

#include "specfem/plugin/injection_plugin.hpp"
#include <gtest/gtest.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace specfem::plugin_sdk_test {

// Owns a dlopen handle and resolves the plugin entry point.
class LoadedPlugin {
public:
  explicit LoadedPlugin(const std::string &path) {
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      error_ = dlerror();
      return;
    }
    const auto entry = reinterpret_cast<specfem_injection_plugin_entry>(
        dlsym(handle_, SPECFEM_INJECTION_PLUGIN_ENTRY_NAME));
    if (entry == nullptr) {
      error_ = "entry point not found";
      return;
    }
    table_ = entry();
  }
  ~LoadedPlugin() {
    if (handle_ != nullptr)
      dlclose(handle_);
  }
  LoadedPlugin(const LoadedPlugin &) = delete;
  LoadedPlugin &operator=(const LoadedPlugin &) = delete;

  const specfem_injection_plugin_table *table() const { return table_; }
  const std::string &error() const { return error_; }

private:
  void *handle_ = nullptr;
  const specfem_injection_plugin_table *table_ = nullptr;
  std::string error_;
};

// Five points in three groups, mirroring what SPECFEM++ sends:
//  - group 0: elastic/isotropic, points 0-2 (x = 1, 2, 3; kappa = 10, 20, 30),
//    velocity in components 0-2 and traction in 3-5;
//  - group 1: elastic/anisotropic with no local points;
//  - group 2: acoustic/isotropic, points 3-4 (x = 4, 5; kappa = 40, 50),
//    displacement in components 0-2 and potential rate in 3.
// Members are public so tests can break individual fields.
struct TestScene {
  TestScene() {
    elastic.struct_size = sizeof(specfem_point_group);
    elastic.medium = specfem_medium_elastic;
    elastic.property = specfem_property_isotropic;
    elastic.nproperties = 3;
    elastic.begin = 0;
    elastic.count = 3;
    elastic.property_names = elastic_names.data();
    elastic.properties = elastic_values.data();
    elastic.ncomponents = 6;
    elastic.nslots = 2;
    elastic.slots = elastic_slots.data();

    anisotropic.struct_size = sizeof(specfem_point_group);
    anisotropic.medium = specfem_medium_elastic;
    anisotropic.property = specfem_property_anisotropic;
    anisotropic.nproperties = 2;
    anisotropic.begin = 3;
    anisotropic.count = 0;
    anisotropic.property_names = anisotropic_names.data();
    anisotropic.ncomponents = 3;
    anisotropic.nslots = 1;
    anisotropic.slots = elastic_slots.data();

    acoustic.struct_size = sizeof(specfem_point_group);
    acoustic.medium = specfem_medium_acoustic;
    acoustic.property = specfem_property_isotropic;
    acoustic.nproperties = 2;
    acoustic.begin = 3;
    acoustic.count = 2;
    acoustic.property_names = acoustic_names.data();
    acoustic.properties = acoustic_values.data();
    acoustic.ncomponents = 4;
    acoustic.nslots = 2;
    acoustic.slots = acoustic_slots.data();

    points.struct_size = sizeof(specfem_points);
    points.ngroups = 3;
    points.n = 5;
    points.x = x.data();
    points.y = zeros.data();
    points.z = zeros.data();
    points.nx = ones.data();
    points.ny = zeros.data();
    points.nz = zeros.data();
    points.groups = groups.data();
  }
  TestScene(const TestScene &) = delete;
  TestScene &operator=(const TestScene &) = delete;

  std::vector<double> x{ 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> zeros = std::vector<double>(5, 0.0);
  std::vector<double> ones = std::vector<double>(5, 1.0);

  std::vector<const char *> elastic_names{ "kappa", "mu", "rho" };
  std::vector<double> elastic_values{ 10.0, 1.0,  2.0, 20.0, 1.0,
                                      2.0,  30.0, 1.0, 2.0 };
  std::vector<specfem_quantity_slot> elastic_slots{
    { specfem_quantity_velocity, 0, 3, 0 },
    { specfem_quantity_traction, 3, 3, 0 }
  };
  std::vector<const char *> anisotropic_names{ "c11", "rho" };
  std::vector<const char *> acoustic_names{ "rho_inverse", "kappa" };
  std::vector<double> acoustic_values{ 0.5, 40.0, 0.5, 50.0 };
  std::vector<specfem_quantity_slot> acoustic_slots{
    { specfem_quantity_displacement, 0, 3, 0 },
    { specfem_quantity_potential_rate, 3, 1, 0 }
  };

  specfem_point_group elastic{};
  specfem_point_group anisotropic{};
  specfem_point_group acoustic{};
  std::vector<const specfem_point_group *> groups{ &elastic, &anisotropic,
                                                   &acoustic };
  specfem_points points{};
};

// Scene with a single elastic/isotropic displacement group of three points.
struct ElasticOnlyScene : TestScene {
  ElasticOnlyScene() {
    elastic.ncomponents = 3;
    elastic.nslots = 1;
    elastic.slots = displacement.data();
    points.n = 3;
    points.ngroups = 1;
  }
  std::vector<specfem_quantity_slot> displacement{
    { specfem_quantity_displacement, 0, 3, 0 }
  };
};

specfem_simulation_info make_info() {
  specfem_simulation_info info{};
  info.struct_size = sizeof(specfem_simulation_info);
  info.dimension = 3;
  info.nstep = 100;
  info.t0 = 0.0;
  info.dt = 0.1;
  return info;
}

specfem_communicator make_communicator() {
  specfem_communicator communicator{};
  communicator.struct_size = sizeof(specfem_communicator);
  communicator.size = 1;
  return communicator;
}

// Contiguous [point][component][step] buffer descriptor over data.
specfem_buffer make_buffer(void *data, const specfem_dtype dtype,
                           const std::int64_t npoints, const int ncomponents,
                           const int nsteps) {
  specfem_buffer buffer{};
  buffer.struct_size = sizeof(specfem_buffer);
  buffer.data = data;
  buffer.dtype = dtype;
  buffer.npoints = npoints;
  buffer.ncomponents = ncomponents;
  buffer.nsteps = nsteps;
  buffer.stride_step = 1;
  buffer.stride_component = nsteps;
  buffer.stride_point = static_cast<std::int64_t>(ncomponents) * nsteps;
  return buffer;
}

// Host storage and descriptors for one window of every group of a TestScene.
template <typename T> struct SceneBuffers {
  SceneBuffers(const specfem_dtype dtype, const int nsteps)
      : elastic(3 * 6 * nsteps, T(-1)), anisotropic(1, T(-1)),
        acoustic(2 * 4 * nsteps, T(-1)),
        descriptors{ make_buffer(elastic.data(), dtype, 3, 6, nsteps),
                     make_buffer(anisotropic.data(), dtype, 0, 3, nsteps),
                     make_buffer(acoustic.data(), dtype, 2, 4, nsteps) },
        pointers{ &descriptors[0], &descriptors[1], &descriptors[2] } {}

  std::vector<T> elastic;
  std::vector<T> anisotropic;
  std::vector<T> acoustic;
  std::vector<specfem_buffer> descriptors;
  std::vector<const specfem_buffer *> pointers;
};

// Number of manifests in a file, and the last one found.
int find_manifests(const std::string &path, specfem_manifest &manifest) {
  std::ifstream file(path, std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
  const char magic[16] = SPECFEM_MANIFEST_MAGIC_INITIALIZER;
  int count = 0;
  auto it = bytes.begin();
  while ((it = std::search(it, bytes.end(), magic, magic + sizeof(magic))) !=
         bytes.end()) {
    if (static_cast<std::size_t>(bytes.end() - it) >= sizeof(specfem_manifest))
      std::memcpy(&manifest, &*it, sizeof(specfem_manifest));
    ++count;
    ++it;
  }
  return count;
}

TEST(PluginSdkPoints, ExposesGroupsPropertiesAndLayout) {
  using specfem::plugin::Medium;
  using specfem::plugin::Property;
  using specfem::plugin::Quantity;
  const TestScene scene;
  const specfem::plugin::Points points(scene.points);

  ASSERT_EQ(points.size(), 5);
  ASSERT_EQ(points.groups().size(), 3u);
  const auto &elastic = points.groups()[0];
  EXPECT_EQ(elastic.medium(), Medium::elastic);
  EXPECT_EQ(elastic.property(), Property::isotropic);
  EXPECT_EQ(elastic.begin(), 0);
  EXPECT_EQ(elastic.size(), 3);
  EXPECT_EQ(elastic.property_names(),
            (std::vector<std::string>{ "kappa", "mu", "rho" }));
  EXPECT_EQ(elastic.property_index("rho"), 2);
  EXPECT_EQ(elastic.property_index("vp"), -1);
  EXPECT_NEAR(elastic.property(1, "kappa"), 20.0, 1e-15);
  EXPECT_NEAR(elastic.property(2, 2), 2.0, 1e-15);
  EXPECT_THROW(elastic.property(0, "vp"), std::out_of_range);
  EXPECT_THROW(elastic.property(3, 0), std::out_of_range);
  EXPECT_EQ(elastic.number_of_components(), 6);
  ASSERT_NE(elastic.slot(Quantity::traction), nullptr);
  EXPECT_EQ(elastic.slot(Quantity::traction)->offset, 3);
  EXPECT_EQ(elastic.slot(Quantity::traction)->count, 3);
  EXPECT_EQ(elastic.slot(Quantity::displacement), nullptr);

  EXPECT_EQ(points.groups()[1].property(), Property::anisotropic);
  EXPECT_EQ(points.groups()[1].size(), 0);

  const auto &acoustic = points.groups()[2];
  EXPECT_EQ(acoustic.medium(), Medium::acoustic);
  EXPECT_EQ(acoustic.begin(), 3);
  EXPECT_NEAR(acoustic.property(1, "kappa"), 50.0, 1e-15);
  ASSERT_NE(acoustic.slot(Quantity::potential_rate), nullptr);
  EXPECT_EQ(acoustic.slot(Quantity::potential_rate)->offset, 3);
  EXPECT_EQ(acoustic.slot(Quantity::potential_rate)->count, 1);
  EXPECT_NEAR(points.x(acoustic.begin() + 1), 5.0, 1e-15);
}

// One way of corrupting a TestScene, and the description used in failures.
struct InvalidPointsCase {
  std::string name;
  std::function<void(TestScene &)> corrupt;
};

void PrintTo(const InvalidPointsCase &value, std::ostream *os) {
  *os << value.name;
}

class PluginSdkInvalidPoints
    : public ::testing::TestWithParam<InvalidPointsCase> {};

TEST_P(PluginSdkInvalidPoints, Rejected) {
  TestScene scene;
  GetParam().corrupt(scene);
  EXPECT_THROW(specfem::plugin::Points{ scene.points }, std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    Cases, PluginSdkInvalidPoints,
    ::testing::Values(
        InvalidPointsCase{ "PointsTooSmall",
                           [](TestScene &s) { s.points.struct_size = 8; } },
        InvalidPointsCase{ "NullCoordinates",
                           [](TestScene &s) { s.points.x = nullptr; } },
        InvalidPointsCase{ "NegativeCount",
                           [](TestScene &s) { s.points.n = -1; } },
        InvalidPointsCase{ "NullGroup",
                           [](TestScene &s) { s.groups[1] = nullptr; } },
        InvalidPointsCase{ "GroupTooSmall",
                           [](TestScene &s) { s.elastic.struct_size = 8; } },
        InvalidPointsCase{ "UnknownMedium",
                           [](TestScene &s) { s.acoustic.medium = 0; } },
        InvalidPointsCase{ "UnknownProperty",
                           [](TestScene &s) { s.acoustic.property = 9; } },
        InvalidPointsCase{ "GroupsOverlap",
                           [](TestScene &s) { s.acoustic.begin = 2; } },
        InvalidPointsCase{ "GroupsLeaveGap",
                           [](TestScene &s) { s.acoustic.count = 1; } },
        InvalidPointsCase{ "GroupOutsidePoints",
                           [](TestScene &s) { s.acoustic.count = 7; } },
        InvalidPointsCase{
            "NullPropertyValues",
            [](TestScene &s) { s.elastic.properties = nullptr; } },
        InvalidPointsCase{ "NullPropertyName",
                           [](TestScene &s) { s.elastic_names[1] = nullptr; } },
        InvalidPointsCase{
            "UnknownQuantity",
            [](TestScene &s) { s.acoustic_slots[0].quantity = 99; } },
        InvalidPointsCase{
            "SlotOutsideComponents",
            [](TestScene &s) { s.acoustic_slots[1].offset = 4; } },
        InvalidPointsCase{
            "EmptySlot", [](TestScene &s) { s.acoustic_slots[1].count = 0; } }),
    [](const ::testing::TestParamInfo<InvalidPointsCase> &info) {
      return info.param.name;
    });

TEST(PluginSdkSimulationInfo, RejectsUnsupportedOrTruncatedInput) {
  auto info = make_info();
  info.dimension = 2;
  EXPECT_THROW(specfem::plugin::SimulationInfo{ info },
               specfem::plugin::Unsupported);

  info = make_info();
  info.struct_size = 16;
  EXPECT_THROW(specfem::plugin::SimulationInfo{ info }, std::invalid_argument);

  auto communicator = make_communicator();
  communicator.struct_size = 4;
  EXPECT_THROW(specfem::plugin::Communicator{ communicator },
               std::invalid_argument);
}

TEST(PluginSdkBuffer, HonoursStridesAndDtype) {
  // Step-major layout: [step][point][component], 2 points x 3 components.
  std::vector<double> values(2 * 3 * 4, -1.0);
  specfem_buffer raw =
      make_buffer(values.data(), specfem_dtype_float64, 2, 3, 4);
  raw.stride_component = 1;
  raw.stride_point = 3;
  raw.stride_step = 6;
  const specfem::plugin::Buffer buffer(raw);
  buffer.set(1, 2, 3, 7.5);
  EXPECT_NEAR(values[3 * 6 + 1 * 3 + 2], 7.5, 1e-15);
  EXPECT_EQ(std::count(values.begin(), values.end(), -1.0),
            static_cast<long>(values.size()) - 1);

  std::vector<float> floats(2 * 3 * 4, 0.0f);
  const specfem::plugin::Buffer float_buffer(
      make_buffer(floats.data(), specfem_dtype_float32, 2, 3, 4));
  float_buffer.set(1, 0, 2, 0.25);
  EXPECT_NEAR(floats[1 * 12 + 0 * 4 + 2], 0.25f, 1e-7f);
}

TEST(PluginSdkBuffer, SetRejectsOutOfRangeIndices) {
  std::vector<double> values(2 * 3 * 4, -1.0);
  const specfem::plugin::Buffer buffer(
      make_buffer(values.data(), specfem_dtype_float64, 2, 3, 4));
  EXPECT_THROW(buffer.set(2, 0, 0, 1.0), std::out_of_range);
  EXPECT_THROW(buffer.set(-1, 0, 0, 1.0), std::out_of_range);
  EXPECT_THROW(buffer.set(0, 3, 0, 1.0), std::out_of_range);
  EXPECT_THROW(buffer.set(0, 0, 4, 1.0), std::out_of_range);
  EXPECT_EQ(std::count(values.begin(), values.end(), -1.0),
            static_cast<long>(values.size()));
}

TEST(PluginSdkBuffer, RejectsInvalidDescriptors) {
  double value = 0.0;
  EXPECT_THROW(specfem::plugin::Buffer{ make_buffer(&value, 7, 1, 1, 1) },
               std::invalid_argument);
  EXPECT_THROW(specfem::plugin::Buffer{ make_buffer(
                   nullptr, specfem_dtype_float64, 1, 1, 1) },
               std::invalid_argument);
  auto truncated = make_buffer(&value, specfem_dtype_float64, 1, 1, 1);
  truncated.struct_size = 8;
  EXPECT_THROW(specfem::plugin::Buffer{ truncated }, std::invalid_argument);
}

TEST(PluginSdkModule, CppPluginTableAndCapabilities) {
  const LoadedPlugin plugin(SPECFEM_TEST_GOOD_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  EXPECT_EQ(table.struct_size, sizeof(specfem_injection_plugin_table));
  EXPECT_EQ(table.abi_version,
            static_cast<uint32_t>(SPECFEM_INJECTION_ABI_VERSION));
  EXPECT_EQ(table.capabilities, 0u);
  EXPECT_STREQ(table.name, "test-good");
  EXPECT_STREQ(table.version, "1.2.3");

  const auto info = make_info();
  const auto communicator = make_communicator();
  void *self = table.create("ok", &info, &communicator);
  ASSERT_NE(self, nullptr) << table.last_error(nullptr);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic,
                           specfem_property_anisotropic,
                           specfem_quantity_velocity),
            1);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic,
                           specfem_property_isotropic,
                           specfem_quantity_potential_rate),
            0);
  EXPECT_EQ(table.supports(self, specfem_medium_acoustic,
                           specfem_property_isotropic,
                           specfem_quantity_potential_rate),
            1);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic_spin,
                           specfem_property_isotropic_cosserat,
                           specfem_quantity_displacement),
            0);
  EXPECT_EQ(table.supports(self, 0, specfem_property_isotropic,
                           specfem_quantity_velocity),
            0);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic, 9,
                           specfem_quantity_velocity),
            0);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic,
                           specfem_property_isotropic, 99),
            0);
  table.destroy(self);
}

TEST(PluginSdkModule, CppPluginFillsEveryGroup) {
  const LoadedPlugin plugin(SPECFEM_TEST_GOOD_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  const auto info = make_info();
  const auto communicator = make_communicator();
  void *self = table.create("ok", &info, &communicator);
  ASSERT_NE(self, nullptr) << table.last_error(nullptr);

  const TestScene scene;
  ASSERT_EQ(table.setup(self, &scene.points), specfem_status_ok)
      << table.last_error(self);

  constexpr int nsteps = 4;
  constexpr int first_step = 5;
  SceneBuffers<double> buffers(specfem_dtype_float64, nsteps);
  ASSERT_EQ(table.fill(self, first_step, 3, buffers.pointers.data()),
            specfem_status_ok)
      << table.last_error(self);

  // base = x + kappa, per group.
  const std::vector<double> elastic_base{ 11.0, 22.0, 33.0 };
  for (int p = 0; p < 3; ++p)
    for (int c = 0; c < 6; ++c)
      for (int s = 0; s < nsteps; ++s)
        EXPECT_NEAR(buffers.elastic[(p * 6 + c) * nsteps + s],
                    1000.0 * (first_step + s) + 10.0 * c + elastic_base[p],
                    1e-12);
  EXPECT_NEAR(buffers.anisotropic[0], -1.0, 1e-15);
  const std::vector<double> acoustic_base{ 44.0, 55.0 };
  for (int p = 0; p < 2; ++p)
    for (int c = 0; c < 4; ++c)
      for (int s = 0; s < nsteps; ++s)
        EXPECT_NEAR(buffers.acoustic[(p * 4 + c) * nsteps + s],
                    1000.0 * (first_step + s) + 10.0 * c + acoustic_base[p],
                    1e-12);

  SceneBuffers<float> floats(specfem_dtype_float32, nsteps);
  ASSERT_EQ(table.fill(self, 0, 3, floats.pointers.data()), specfem_status_ok);
  EXPECT_NEAR(floats.acoustic[(1 * 4 + 3) * nsteps + 3],
              3000.0f + 30.0f + 55.0f, 1e-3f);
  table.destroy(self);
}

TEST(PluginSdkModule, FillChecksBuffersAgainstSetup) {
  const LoadedPlugin plugin(SPECFEM_TEST_GOOD_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  const auto info = make_info();
  const auto communicator = make_communicator();
  void *self = table.create("ok", &info, &communicator);
  ASSERT_NE(self, nullptr) << table.last_error(nullptr);

  SceneBuffers<double> buffers(specfem_dtype_float64, 2);
  EXPECT_EQ(table.fill(self, 0, 3, buffers.pointers.data()),
            specfem_status_error);
  EXPECT_STREQ(table.last_error(self), "fill: setup has not succeeded");

  const TestScene scene;
  ASSERT_EQ(table.setup(self, &scene.points), specfem_status_ok);

  EXPECT_EQ(table.fill(self, 0, 2, buffers.pointers.data()),
            specfem_status_error);
  EXPECT_STREQ(table.last_error(self), "fill: expected one buffer per group");

  buffers.descriptors[2].npoints = 3;
  EXPECT_EQ(table.fill(self, 0, 3, buffers.pointers.data()),
            specfem_status_error);
  EXPECT_STREQ(table.last_error(self),
               "fill: buffer 2 does not match its group");
  buffers.descriptors[2].npoints = 2;

  buffers.descriptors[0].ncomponents = 5;
  EXPECT_EQ(table.fill(self, 0, 3, buffers.pointers.data()),
            specfem_status_error);
  buffers.descriptors[0].ncomponents = 6;

  buffers.descriptors[1].nsteps = 1;
  EXPECT_EQ(table.fill(self, 0, 3, buffers.pointers.data()),
            specfem_status_error);
  EXPECT_STREQ(table.last_error(self), "fill: buffers differ in step count");
  buffers.descriptors[1].nsteps = 2;

  EXPECT_EQ(table.fill(self, 99, 3, buffers.pointers.data()),
            specfem_status_error);
  EXPECT_STREQ(table.last_error(self),
               "fill: window outside the simulation steps");

  EXPECT_EQ(table.fill(self, 98, 3, buffers.pointers.data()), specfem_status_ok)
      << table.last_error(self);
  EXPECT_STREQ(table.last_error(self), "");
  table.destroy(self);
}

TEST(PluginSdkModule, ExceptionsBecomeStatusAndMessage) {
  const LoadedPlugin plugin(SPECFEM_TEST_GOOD_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  const auto info = make_info();
  const auto communicator = make_communicator();

  EXPECT_EQ(table.create("throw-in-create", &info, &communicator), nullptr);
  EXPECT_STREQ(table.last_error(nullptr), "create failed on purpose");
  EXPECT_EQ(table.create(nullptr, &info, &communicator), nullptr);
  EXPECT_STREQ(table.last_error(nullptr), "create: null argument");

  auto info_2d = make_info();
  info_2d.dimension = 2;
  EXPECT_EQ(table.create("ok", &info_2d, &communicator), nullptr);
  EXPECT_NE(std::string(table.last_error(nullptr)).find("dimension 3"),
            std::string::npos);

  const TestScene scene;

  void *setup_fails = table.create("throw-in-setup", &info, &communicator);
  ASSERT_NE(setup_fails, nullptr);
  EXPECT_EQ(table.setup(setup_fails, &scene.points), specfem_status_error);
  EXPECT_STREQ(table.last_error(setup_fails), "setup failed on purpose");
  EXPECT_EQ(table.setup(setup_fails, nullptr), specfem_status_error);
  EXPECT_STREQ(table.last_error(setup_fails), "setup: null argument");
  table.destroy(setup_fails);

  void *unsupported =
      table.create("unsupported-in-setup", &info, &communicator);
  ASSERT_NE(unsupported, nullptr);
  EXPECT_EQ(table.setup(unsupported, &scene.points),
            specfem_status_unsupported);
  EXPECT_STREQ(table.last_error(unsupported), "setup cannot serve this run");
  table.destroy(unsupported);

  void *fill_fails = table.create("throw-in-fill", &info, &communicator);
  ASSERT_NE(fill_fails, nullptr);
  ASSERT_EQ(table.setup(fill_fails, &scene.points), specfem_status_ok);
  SceneBuffers<double> buffers(specfem_dtype_float64, 1);
  EXPECT_EQ(table.fill(fill_fails, 0, 3, buffers.pointers.data()),
            specfem_status_error);
  EXPECT_STREQ(table.last_error(fill_fails), "fill failed on purpose");
  table.destroy(fill_fails);

  void *out_of_range = table.create("write-out-of-range", &info, &communicator);
  ASSERT_NE(out_of_range, nullptr);
  ASSERT_EQ(table.setup(out_of_range, &scene.points), specfem_status_ok);
  EXPECT_EQ(table.fill(out_of_range, 0, 3, buffers.pointers.data()),
            specfem_status_error);
  EXPECT_STREQ(table.last_error(out_of_range), "buffer: index out of range");
  table.destroy(out_of_range);

  table.destroy(nullptr);
}

// Reporting an error must not allocate: out-of-memory and oversized messages
// still produce a status and a (possibly truncated) message.
TEST(PluginSdkModule, ErrorReportingSurvivesAllocationFailure) {
  const LoadedPlugin plugin(SPECFEM_TEST_GOOD_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  const auto info = make_info();
  const auto communicator = make_communicator();

  EXPECT_EQ(table.create("bad-alloc-in-create", &info, &communicator), nullptr);
  EXPECT_STRNE(table.last_error(nullptr), "");

  const TestScene scene;
  SceneBuffers<double> buffers(specfem_dtype_float64, 1);

  void *bad_alloc = table.create("bad-alloc-in-fill", &info, &communicator);
  ASSERT_NE(bad_alloc, nullptr);
  ASSERT_EQ(table.setup(bad_alloc, &scene.points), specfem_status_ok);
  EXPECT_EQ(table.fill(bad_alloc, 0, 3, buffers.pointers.data()),
            specfem_status_error);
  EXPECT_STRNE(table.last_error(bad_alloc), "");
  table.destroy(bad_alloc);

  void *long_error = table.create("long-error-in-fill", &info, &communicator);
  ASSERT_NE(long_error, nullptr);
  ASSERT_EQ(table.setup(long_error, &scene.points), specfem_status_ok);
  EXPECT_EQ(table.fill(long_error, 0, 3, buffers.pointers.data()),
            specfem_status_error);
  EXPECT_EQ(std::string(table.last_error(long_error)),
            std::string(specfem::plugin_impl::error_capacity - 1, 'e'));
  table.destroy(long_error);
}

TEST(PluginSdkModule, PureCPlugin) {
  const LoadedPlugin plugin(SPECFEM_TEST_C_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  EXPECT_STREQ(table.name, "test-c");

  const auto info = make_info();
  const auto communicator = make_communicator();
  void *self = table.create("unused", &info, &communicator);
  ASSERT_NE(self, nullptr);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic,
                           specfem_property_isotropic,
                           specfem_quantity_displacement),
            1);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic,
                           specfem_property_anisotropic,
                           specfem_quantity_displacement),
            0);

  const TestScene mixed;
  EXPECT_EQ(table.setup(self, &mixed.points), specfem_status_unsupported);
  EXPECT_STREQ(table.last_error(self), "only elastic isotropic displacement");

  const ElasticOnlyScene scene;
  ASSERT_EQ(table.setup(self, &scene.points), specfem_status_ok);

  std::vector<double> values(3 * 3 * 2, 0.0);
  const auto buffer =
      make_buffer(values.data(), specfem_dtype_float64, 3, 3, 2);
  const specfem_buffer *buffers[] = { &buffer };
  ASSERT_EQ(table.fill(self, 10, 1, buffers), specfem_status_ok);
  EXPECT_NEAR(values[0], 52.0, 1e-12);
  EXPECT_NEAR(values[values.size() - 1], 53.0, 1e-12);
  table.destroy(self);
}

TEST(PluginSdkManifest, ReadableWithoutLoading) {
  for (const auto &[path, name, version] :
       { std::tuple<std::string, std::string, std::string>{
             SPECFEM_TEST_GOOD_PLUGIN, "test-good", "1.2.3" },
         std::tuple<std::string, std::string, std::string>{
             SPECFEM_TEST_C_PLUGIN, "test-c", "0.1.0" } }) {
    specfem_manifest manifest{};
    EXPECT_EQ(find_manifests(path, manifest), 1) << path;
    EXPECT_EQ(manifest.manifest_version,
              static_cast<uint32_t>(SPECFEM_MANIFEST_VERSION));
    EXPECT_EQ(manifest.abi_version,
              static_cast<uint32_t>(SPECFEM_INJECTION_ABI_VERSION));
    EXPECT_EQ(manifest.plugin_kind,
              static_cast<uint32_t>(SPECFEM_PLUGIN_KIND_INJECTION));
    EXPECT_EQ(std::string(manifest.name), name);
    EXPECT_EQ(std::string(manifest.version), version);
  }
}

} // namespace specfem::plugin_sdk_test
