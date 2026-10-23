// Tests for the injection plugin SDK: the C++ helper types, and real plugin
// modules loaded with dlopen exactly as SPECFEM++ will load them.

#include "specfem/plugin/injection_plugin.hpp"
#include <gtest/gtest.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
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
    const auto entry = reinterpret_cast<spfp_injection_plugin_entry>(
        dlsym(handle_, SPFP_INJECTION_PLUGIN_ENTRY_NAME));
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

  const spfp_injection_plugin *table() const { return table_; }
  const std::string &error() const { return error_; }

private:
  void *handle_ = nullptr;
  const spfp_injection_plugin *table_ = nullptr;
  std::string error_;
};

// Three elastic points with x = 1, 2, 3.
class TestPoints {
public:
  TestPoints() {
    points_.n = static_cast<std::int64_t>(x_.size());
    points_.x = x_.data();
    points_.y = zeros_.data();
    points_.z = zeros_.data();
    points_.nx = ones_.data();
    points_.ny = zeros_.data();
    points_.nz = zeros_.data();
    points_.medium = medium_.data();
    points_.rho = ones_.data();
    points_.vp = ones_.data();
    points_.vs = ones_.data();
  }
  const spfp_points &get() const { return points_; }

private:
  std::vector<double> x_{ 1.0, 2.0, 3.0 };
  std::vector<double> zeros_ = std::vector<double>(3, 0.0);
  std::vector<double> ones_ = std::vector<double>(3, 1.0);
  std::vector<std::int32_t> medium_ =
      std::vector<std::int32_t>(3, spfp_medium_elastic);
  spfp_points points_{};
};

spfp_simulation_info make_info() {
  spfp_simulation_info info{};
  info.dimension = 3;
  info.nstep = 100;
  info.t0 = 0.0;
  info.dt = 0.1;
  return info;
}

spfp_communicator make_communicator() {
  spfp_communicator communicator{};
  communicator.size = 1;
  return communicator;
}

// Contiguous [point][component][step] buffer descriptor over data.
spfp_buffer make_buffer(void *data, const spfp_dtype dtype,
                        const std::int64_t npoints, const int ncomponents,
                        const int nsteps) {
  spfp_buffer buffer{};
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

// Number of manifests in a file, and the last one found.
int find_manifests(const std::string &path, spfp_manifest &manifest) {
  std::ifstream file(path, std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
  const char magic[16] = SPFP_MANIFEST_MAGIC_INITIALIZER;
  int count = 0;
  auto it = bytes.begin();
  while ((it = std::search(it, bytes.end(), magic, magic + sizeof(magic))) !=
         bytes.end()) {
    if (static_cast<std::size_t>(bytes.end() - it) >= sizeof(spfp_manifest))
      std::memcpy(&manifest, &*it, sizeof(spfp_manifest));
    ++count;
    ++it;
  }
  return count;
}

TEST(PluginSdkRequest, SlotOffsetsFollowRequestOrder) {
  using specfem::plugin::Medium;
  using specfem::plugin::Quantity;
  const specfem::plugin::Request request(
      { Quantity::velocity, Quantity::traction },
      { Quantity::displacement, Quantity::potential_rate });

  EXPECT_EQ(request.component_count(Medium::elastic), 6);
  EXPECT_EQ(request.slot_offset(Medium::elastic, Quantity::velocity), 0);
  EXPECT_EQ(request.slot_offset(Medium::elastic, Quantity::traction), 3);
  EXPECT_EQ(request.slot_offset(Medium::elastic, Quantity::displacement), -1);

  EXPECT_EQ(request.component_count(Medium::acoustic), 4);
  EXPECT_EQ(request.slot_offset(Medium::acoustic, Quantity::potential_rate), 3);
}

TEST(PluginSdkRequest, RejectsInvalidLists) {
  spfp_request null_list{};
  null_list.n_elastic = 1;
  EXPECT_THROW(specfem::plugin::Request{ null_list }, std::invalid_argument);

  const spfp_quantity unknown[] = { 99 };
  spfp_request bad_quantity{};
  bad_quantity.n_elastic = 1;
  bad_quantity.elastic = unknown;
  EXPECT_THROW(specfem::plugin::Request{ bad_quantity }, std::invalid_argument);
}

TEST(PluginSdkBuffer, HonoursStridesAndDtype) {
  // Step-major layout: [step][point][component], 2 points x 3 components.
  std::vector<double> values(2 * 3 * 4, -1.0);
  spfp_buffer raw = make_buffer(values.data(), spfp_dtype_float64, 2, 3, 4);
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
      make_buffer(floats.data(), spfp_dtype_float32, 2, 3, 4));
  float_buffer.set(1, 0, 2, 0.25);
  EXPECT_NEAR(floats[1 * 12 + 0 * 4 + 2], 0.25f, 1e-7f);
}

TEST(PluginSdkBuffer, RejectsUnknownDtype) {
  double value = 0.0;
  spfp_buffer raw = make_buffer(&value, 7, 1, 1, 1);
  EXPECT_THROW(specfem::plugin::Buffer{ raw }, std::invalid_argument);
}

TEST(PluginSdkModule, CppPluginTableAndCapabilities) {
  const LoadedPlugin plugin(SPFP_TEST_GOOD_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  EXPECT_EQ(table.abi_version,
            static_cast<uint32_t>(SPFP_INJECTION_ABI_VERSION));
  EXPECT_EQ(table.capabilities, 0u);
  EXPECT_STREQ(table.name, "test-good");
  EXPECT_STREQ(table.version, "1.2.3");

  const auto info = make_info();
  const auto communicator = make_communicator();
  void *self = table.create("ok", &info, &communicator);
  ASSERT_NE(self, nullptr) << table.last_error(nullptr);
  EXPECT_EQ(table.supports(self, spfp_medium_elastic, spfp_quantity_velocity),
            1);
  EXPECT_EQ(table.supports(self, spfp_medium_acoustic, spfp_quantity_velocity),
            0);
  EXPECT_EQ(
      table.supports(self, spfp_medium_elastic, spfp_quantity_potential_rate),
      0);
  EXPECT_EQ(table.supports(self, 7, spfp_quantity_velocity), 0);
  EXPECT_EQ(table.supports(self, spfp_medium_elastic, 99), 0);
  table.destroy(self);
}

TEST(PluginSdkModule, CppPluginFillsRequestedWindow) {
  const LoadedPlugin plugin(SPFP_TEST_GOOD_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  const auto info = make_info();
  const auto communicator = make_communicator();
  void *self = table.create("ok", &info, &communicator);
  ASSERT_NE(self, nullptr) << table.last_error(nullptr);

  const TestPoints points;
  const spfp_quantity elastic[] = { spfp_quantity_velocity,
                                    spfp_quantity_traction };
  spfp_request request{};
  request.n_elastic = 2;
  request.elastic = elastic;
  ASSERT_EQ(table.setup(self, &points.get(), &request), spfp_status_ok)
      << table.last_error(self);

  constexpr int ncomponents = 6;
  constexpr int nsteps = 4;
  constexpr int first_step = 5;
  std::vector<double> values(3 * ncomponents * nsteps, -1.0);
  const auto buffer =
      make_buffer(values.data(), spfp_dtype_float64, 3, ncomponents, nsteps);
  ASSERT_EQ(table.fill(self, first_step, &buffer), spfp_status_ok)
      << table.last_error(self);
  for (int point = 0; point < 3; ++point)
    for (int component = 0; component < ncomponents; ++component)
      for (int step = 0; step < nsteps; ++step)
        EXPECT_NEAR(
            values[point * ncomponents * nsteps + component * nsteps + step],
            1000.0 * (first_step + step) + 10.0 * component + (point + 1.0),
            1e-12);

  std::vector<float> floats(3 * ncomponents * nsteps, -1.0f);
  const auto float_buffer =
      make_buffer(floats.data(), spfp_dtype_float32, 3, ncomponents, nsteps);
  ASSERT_EQ(table.fill(self, 0, &float_buffer), spfp_status_ok);
  EXPECT_NEAR(floats[2 * ncomponents * nsteps + 5 * nsteps + 3],
              3000.0f + 50.0f + 3.0f, 1e-3f);
  table.destroy(self);
}

TEST(PluginSdkModule, ExceptionsBecomeStatusAndMessage) {
  const LoadedPlugin plugin(SPFP_TEST_GOOD_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  const auto info = make_info();
  const auto communicator = make_communicator();

  EXPECT_EQ(table.create("throw-in-create", &info, &communicator), nullptr);
  EXPECT_STREQ(table.last_error(nullptr), "create failed on purpose");
  EXPECT_EQ(table.create(nullptr, &info, &communicator), nullptr);
  EXPECT_STREQ(table.last_error(nullptr), "create: null argument");

  const TestPoints points;
  const spfp_quantity elastic[] = { spfp_quantity_velocity };
  spfp_request request{};
  request.n_elastic = 1;
  request.elastic = elastic;

  void *setup_fails = table.create("throw-in-setup", &info, &communicator);
  ASSERT_NE(setup_fails, nullptr);
  EXPECT_EQ(table.setup(setup_fails, &points.get(), &request),
            spfp_status_error);
  EXPECT_STREQ(table.last_error(setup_fails), "setup failed on purpose");
  EXPECT_EQ(table.setup(setup_fails, nullptr, &request), spfp_status_error);
  EXPECT_STREQ(table.last_error(setup_fails), "setup: null argument");
  table.destroy(setup_fails);

  void *fill_fails = table.create("throw-in-fill", &info, &communicator);
  ASSERT_NE(fill_fails, nullptr);
  ASSERT_EQ(table.setup(fill_fails, &points.get(), &request), spfp_status_ok);
  std::vector<double> values(3 * 3, 0.0);
  const auto buffer = make_buffer(values.data(), spfp_dtype_float64, 3, 3, 1);
  EXPECT_EQ(table.fill(fill_fails, 0, &buffer), spfp_status_error);
  EXPECT_STREQ(table.last_error(fill_fails), "fill failed on purpose");
  table.destroy(fill_fails);

  table.destroy(nullptr);
}

TEST(PluginSdkModule, PureCPlugin) {
  const LoadedPlugin plugin(SPFP_TEST_C_PLUGIN);
  ASSERT_NE(plugin.table(), nullptr) << plugin.error();
  const auto &table = *plugin.table();
  EXPECT_STREQ(table.name, "test-c");

  const auto info = make_info();
  const auto communicator = make_communicator();
  void *self = table.create("unused", &info, &communicator);
  ASSERT_NE(self, nullptr);
  EXPECT_EQ(
      table.supports(self, spfp_medium_elastic, spfp_quantity_displacement), 1);
  EXPECT_EQ(table.supports(self, spfp_medium_elastic, spfp_quantity_traction),
            0);

  const TestPoints points;
  const spfp_quantity elastic[] = { spfp_quantity_displacement };
  spfp_request request{};
  request.n_elastic = 1;
  request.elastic = elastic;
  ASSERT_EQ(table.setup(self, &points.get(), &request), spfp_status_ok);

  std::vector<double> values(3 * 3 * 2, 0.0);
  const auto buffer = make_buffer(values.data(), spfp_dtype_float64, 3, 3, 2);
  ASSERT_EQ(table.fill(self, 10, &buffer), spfp_status_ok);
  EXPECT_NEAR(values[0], 52.0, 1e-12);
  EXPECT_NEAR(values[values.size() - 1], 53.0, 1e-12);
  table.destroy(self);
}

TEST(PluginSdkManifest, ReadableWithoutLoading) {
  for (const auto &[path, name, version] :
       { std::tuple<std::string, std::string, std::string>{
             SPFP_TEST_GOOD_PLUGIN, "test-good", "1.2.3" },
         std::tuple<std::string, std::string, std::string>{
             SPFP_TEST_C_PLUGIN, "test-c", "0.1.0" } }) {
    spfp_manifest manifest{};
    EXPECT_EQ(find_manifests(path, manifest), 1) << path;
    EXPECT_EQ(manifest.manifest_version,
              static_cast<uint32_t>(SPFP_MANIFEST_VERSION));
    EXPECT_EQ(manifest.abi_version,
              static_cast<uint32_t>(SPFP_INJECTION_ABI_VERSION));
    EXPECT_EQ(manifest.plugin_kind,
              static_cast<uint32_t>(SPFP_PLUGIN_KIND_INJECTION));
    EXPECT_EQ(std::string(manifest.name), name);
    EXPECT_EQ(std::string(manifest.version), version);
  }
}

} // namespace specfem::plugin_sdk_test
