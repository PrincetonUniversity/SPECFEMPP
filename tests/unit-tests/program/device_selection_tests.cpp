#include "specfem/program/context.hpp"
#include "specfem/program/device_selection.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

// Each case runs in its own re-executed process: Kokkos can be initialized
// once per process, and Metal must not be initialized in a fork()ed child.
// The "threadsafe" style re-executes the test binary instead of forking.

#if defined(SPECFEM_ENABLE_METAL)

namespace specfem::program_tests_impl {
// Exit codes reported by the child process.
constexpr int selected_metal = 0;
constexpr int rejected_device = 3;
constexpr int unexpected = 4;

[[noreturn]] void run_context(const std::vector<std::string> &args) {
  int code = unexpected;
  try {
    // Destroy the context (finalizing Kokkos) before std::exit.
    specfem::program::Context context(args);
    const std::string device = specfem::program_impl::check_execution_device();
    code =
        device.find("Metal") != std::string::npos ? selected_metal : unexpected;
  } catch (const std::runtime_error &) {
    code = rejected_device;
  }
  std::exit(code);
}
} // namespace specfem::program_tests_impl

TEST(DeviceSelection, DefaultsToMetalGpu) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        ::unsetenv("KOKKOS_DEVICE_ID");
        ::unsetenv("KOKKOS_VISIBLE_DEVICES");
        ::unsetenv("KOKKOS_MAP_DEVICE_ID_BY");
        specfem::program_tests_impl::run_context({ "device_selection_tests" });
      },
      ::testing::ExitedWithCode(specfem::program_tests_impl::selected_metal),
      "");
}

TEST(DeviceSelection, RejectsHostDevice) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  // Device 0 is AdaptiveCpp's OpenMP host device when it is visible.
  EXPECT_EXIT(
      {
        ::setenv("ACPP_VISIBILITY_MASK", "omp;metal", 1);
        specfem::program_tests_impl::run_context(
            { "device_selection_tests", "--kokkos-device-id=0" });
      },
      ::testing::ExitedWithCode(specfem::program_tests_impl::rejected_device),
      "");
}

#else

TEST(DeviceSelection, DefaultsToMetalGpu) {
  GTEST_SKIP() << "Only meaningful with SPECFEM_ENABLE_METAL";
}

TEST(DeviceSelection, NoCheckOnOtherBuilds) {
  EXPECT_TRUE(specfem::program_impl::check_execution_device().empty());
}

#endif
