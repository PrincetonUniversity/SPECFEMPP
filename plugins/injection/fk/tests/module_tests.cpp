// Loads the FK plugin module with dlopen and drives it through the ABI, as
// SPECFEM++ does. The module owns a private Kokkos, so this also exercises its
// initialize/finalize lifecycle.

#include "specfem/plugin/injection_abi.h"
#include <gtest/gtest.h>

#include <dlfcn.h>

#include <string>

namespace specfem::fk_plugin_test {

specfem_simulation_info make_info() {
  specfem_simulation_info info{};
  info.struct_size = sizeof(specfem_simulation_info);
  info.dimension = 3;
  info.nstep = 10;
  info.dt = 0.1;
  return info;
}

specfem_communicator make_communicator() {
  specfem_communicator communicator{};
  communicator.struct_size = sizeof(specfem_communicator);
  communicator.size = 1;
  return communicator;
}

} // namespace specfem::fk_plugin_test

// One test: the module's Kokkos state persists for the life of the process,
// so the steps must run in this order.
TEST(FkPluginModule, SkeletonLifecycleThroughAbi) {
  void *library = dlopen(FK_PLUGIN_MODULE, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(library, nullptr) << dlerror();
  const auto entry = reinterpret_cast<specfem_injection_plugin_entry>(
      dlsym(library, SPECFEM_INJECTION_PLUGIN_ENTRY_NAME));
  ASSERT_NE(entry, nullptr);
  const specfem_injection_plugin_table &table = *entry();
  EXPECT_EQ(table.abi_version, SPECFEM_INJECTION_ABI_VERSION);
  EXPECT_EQ(std::string(table.name), "fk");

  const auto info = specfem::fk_plugin_test::make_info();
  const auto communicator = specfem::fk_plugin_test::make_communicator();

  void *self = table.create("unused.yaml", &info, &communicator);
  ASSERT_NE(self, nullptr) << table.last_error(nullptr);
  EXPECT_EQ(table.supports(self, specfem_medium_elastic,
                           specfem_property_isotropic,
                           specfem_quantity_velocity),
            0);

  specfem_points points{};
  points.struct_size = sizeof(specfem_points);
  EXPECT_EQ(table.setup(self, &points), specfem_status_unsupported);
  EXPECT_NE(std::string(table.last_error(self)).find("not implemented"),
            std::string::npos);
  table.destroy(self);

  // The last destroy finalized the module's Kokkos, which cannot restart.
  EXPECT_EQ(table.create("unused.yaml", &info, &communicator), nullptr);
  EXPECT_NE(std::string(table.last_error(nullptr)).find("already finalized"),
            std::string::npos);

  dlclose(library);
}
