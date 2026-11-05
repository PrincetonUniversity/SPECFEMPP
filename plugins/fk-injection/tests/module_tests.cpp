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
  // The mesh box holds every point.
  info.bounds[0] = info.bounds[2] = -10000;
  info.bounds[1] = info.bounds[3] = 10000;
  info.bounds[4] = -100000;
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
  - acoustic: { rho: 1000, vp: 1500, thickness: 10000 }
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

  // An elastic point 40 km down requesting velocity and traction, and an
  // acoustic point 5 km down requesting potential rate and displacement.
  const double rho = 2700;
  const double mu = rho * 1732.051 * 1732.051;
  const double kappa = rho * 3000.0 * 3000.0 - 4.0 * mu / 3.0;
  const std::vector<double> properties = { kappa, mu, rho };
  const char *const names[] = { "kappa", "mu", "rho" };
  const specfem_quantity_slot slots[] = {
    { specfem_quantity_velocity, 0, 3, 0 },
    { specfem_quantity_traction, 3, 3, 0 }
  };
  specfem_point_group group{};
  group.struct_size = sizeof(specfem_point_group);
  group.medium = specfem_medium_elastic;
  group.property = specfem_property_isotropic;
  group.nproperties = 3;
  group.count = 1;
  group.property_names = names;
  group.properties = properties.data();
  group.ncomponents = 6;
  group.nslots = 2;
  group.slots = slots;

  const std::vector<double> fluid_properties = { 1.0 / 1000.0,
                                                 1000.0 * 1500.0 * 1500.0 };
  const char *const fluid_names[] = { "rho_inverse", "kappa" };
  const specfem_quantity_slot fluid_slots[] = {
    { specfem_quantity_potential_rate, 0, 1, 0 },
    { specfem_quantity_displacement, 1, 3, 0 }
  };
  specfem_point_group fluid_group{};
  fluid_group.struct_size = sizeof(specfem_point_group);
  fluid_group.medium = specfem_medium_acoustic;
  fluid_group.property = specfem_property_isotropic;
  fluid_group.nproperties = 2;
  fluid_group.begin = 1;
  fluid_group.count = 1;
  fluid_group.property_names = fluid_names;
  fluid_group.properties = fluid_properties.data();
  fluid_group.ncomponents = 4;
  fluid_group.nslots = 2;
  fluid_group.slots = fluid_slots;

  const specfem_point_group *const groups[] = { &group, &fluid_group };
  const double x[] = { 0, 1000 }, y[] = { 0, -2000 }, z[] = { -40000, -5000 };
  const double nx[] = { 0, 0 }, ny[] = { 0, 0 }, nz[] = { -1, 1 };
  specfem_points points{};
  points.struct_size = sizeof(specfem_points);
  points.ngroups = 2;
  points.n = 2;
  points.x = x;
  points.y = y;
  points.z = z;
  points.nx = nx;
  points.ny = ny;
  points.nz = nz;
  points.groups = groups;

  ASSERT_EQ(table.setup(self, &points), specfem_status_ok)
      << table.last_error(self);

  // Row-major (component, step) per point.
  const auto make_buffer = [&](std::vector<double> &values,
                               const int ncomponents) {
    values.assign(static_cast<std::size_t>(ncomponents) * info.nstep, 0.0);
    specfem_buffer buffer{};
    buffer.struct_size = sizeof(specfem_buffer);
    buffer.dtype = specfem_dtype_float64;
    buffer.data = values.data();
    buffer.npoints = 1;
    buffer.ncomponents = ncomponents;
    buffer.nsteps = info.nstep;
    buffer.stride_point = static_cast<std::int64_t>(ncomponents) * info.nstep;
    buffer.stride_component = info.nstep;
    buffer.stride_step = 1;
    return buffer;
  };
  std::vector<double> solid_values;
  std::vector<double> fluid_values;
  const specfem_buffer solid_buffer = make_buffer(solid_values, 6);
  const specfem_buffer fluid_buffer = make_buffer(fluid_values, 4);
  const specfem_buffer *const buffers[] = { &solid_buffer, &fluid_buffer };
  ASSERT_EQ(table.fill(self, 0, 2, buffers), specfem_status_ok)
      << table.last_error(self);

  // Peak of components [first, first + count) of one buffer.
  const auto peak = [&](const std::vector<double> &values, const int first,
                        const int count) {
    double result = 0.0;
    for (std::size_t i = static_cast<std::size_t>(first) * info.nstep;
         i < static_cast<std::size_t>(first + count) * info.nstep; ++i)
      result = std::max(result, std::fabs(values[i]));
    return result;
  };
  EXPECT_GT(peak(solid_values, 0, 3), 0.0) << "velocity";
  EXPECT_GT(peak(solid_values, 3, 3), 0.0) << "traction";
  EXPECT_GT(peak(fluid_values, 0, 1), 0.0) << "potential rate";
  EXPECT_GT(peak(fluid_values, 1, 3), 0.0) << "displacement";
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
