// Loads the FK plugin module with dlopen and drives it through the ABI, as
// SPECFEM++ does.

#include "specfem/plugin/injection_abi.h"
#include <gtest/gtest.h>

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace fk_injection_test {

specfem_simulation_info make_info() {
  specfem_simulation_info info{};
  info.struct_size = sizeof(specfem_simulation_info);
  info.dimension = 3;
  info.nstep = 600;
  info.dt = 0.15;
  return info;
}

specfem_communicator make_communicator() {
  specfem_communicator communicator{};
  communicator.struct_size = sizeof(specfem_communicator);
  communicator.size = 1;
  return communicator;
}

constexpr const char *config_path = "fk_module_test.yaml";

void write_config() {
  std::ofstream(config_path) << R"yaml(
layers:
  - elastic: { rho: 2700, vp: 3000, vs: 1732.051, thickness: 0 }
incidence:
  type: P
  back-azimuth: 0
  take-off: 20
  origin: [0, 0, -224000]
time-window: { frequency-max: 0.2, frequency-sampling: 2, length: 180 }
)yaml";
}

} // namespace fk_injection_test

TEST(FkPluginModule, LifecycleThroughAbi) {
  void *library = dlopen(FK_INJECTION_MODULE, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(library, nullptr) << dlerror();
  const auto entry = reinterpret_cast<specfem_injection_plugin_entry>(
      dlsym(library, SPECFEM_INJECTION_PLUGIN_ENTRY_NAME));
  ASSERT_NE(entry, nullptr);
  const specfem_injection_plugin_table &table = *entry();
  EXPECT_EQ(table.abi_version, SPECFEM_INJECTION_ABI_VERSION);
  EXPECT_EQ(std::string(table.name), "fk");

  const auto info = fk_injection_test::make_info();
  const auto communicator = fk_injection_test::make_communicator();

  EXPECT_EQ(table.create("no_such_file.yaml", &info, &communicator), nullptr);
  EXPECT_NE(std::string(table.last_error(nullptr)).find("no_such_file.yaml"),
            std::string::npos);

  fk_injection_test::write_config();
  void *self =
      table.create(fk_injection_test::config_path, &info, &communicator);
  ASSERT_NE(self, nullptr) << table.last_error(nullptr);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic,
                           specfem_property_isotropic,
                           specfem_quantity_velocity),
            1);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic,
                           specfem_property_isotropic,
                           specfem_quantity_acceleration),
            0);

  // One elastic point, 40 km down, requesting velocity.
  const double rho = 2700;
  const double mu = rho * 1732.051 * 1732.051;
  const double kappa = rho * 3000.0 * 3000.0 - 4.0 * mu / 3.0;
  const std::vector<double> properties = { kappa, mu, rho };
  const char *const names[] = { "kappa", "mu", "rho" };
  const specfem_quantity_slot slot = { specfem_quantity_velocity, 0, 3, 0 };
  specfem_point_group group{};
  group.struct_size = sizeof(specfem_point_group);
  group.medium = specfem_medium_elastic;
  group.property = specfem_property_isotropic;
  group.nproperties = 3;
  group.count = 1;
  group.property_names = names;
  group.properties = properties.data();
  group.ncomponents = 3;
  group.nslots = 1;
  group.slots = &slot;
  const specfem_point_group *const groups[] = { &group };
  const double x = 0, y = 0, z = -40000, nx = 0, ny = 0, nz = -1;
  specfem_points points{};
  points.struct_size = sizeof(specfem_points);
  points.ngroups = 1;
  points.n = 1;
  points.x = &x;
  points.y = &y;
  points.z = &z;
  points.nx = &nx;
  points.ny = &ny;
  points.nz = &nz;
  points.groups = groups;

  ASSERT_EQ(table.setup(self, &points), specfem_status_ok)
      << table.last_error(self);

  std::vector<double> values(3 * info.nstep, 0.0);
  specfem_buffer buffer{};
  buffer.struct_size = sizeof(specfem_buffer);
  buffer.dtype = specfem_dtype_float64;
  buffer.data = values.data();
  buffer.npoints = 1;
  buffer.ncomponents = 3;
  buffer.nsteps = info.nstep;
  buffer.stride_point = 3 * info.nstep;
  buffer.stride_component = info.nstep;
  buffer.stride_step = 1;
  const specfem_buffer *const buffers[] = { &buffer };
  ASSERT_EQ(table.fill(self, 0, 1, buffers), specfem_status_ok)
      << table.last_error(self);
  double peak = 0.0;
  for (const double value : values)
    peak = std::max(peak, std::fabs(value));
  EXPECT_GT(peak, 0.0);
  table.destroy(self);

  // No process-wide state, and a wrong material is unsupported, not an error.
  void *second =
      table.create(fk_injection_test::config_path, &info, &communicator);
  ASSERT_NE(second, nullptr) << table.last_error(nullptr);
  const std::vector<double> wrong = { kappa, mu, 2 * rho };
  group.properties = wrong.data();
  EXPECT_EQ(table.setup(second, &points), specfem_status_unsupported);
  EXPECT_NE(std::string(table.last_error(second)).find("(rho, vp, vs)"),
            std::string::npos)
      << table.last_error(second);
  table.destroy(second);

  std::remove(fk_injection_test::config_path);
  dlclose(library);
}
