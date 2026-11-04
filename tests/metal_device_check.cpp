#include "specfem/program/device_selection.hpp"
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <gtest/gtest.h>
#include <string>

// Linked into every test executable of an AdaptiveCpp/Metal build by
// specfem_add_test() (tests/test_registration.cmake). Many tests initialize
// Kokkos directly instead of through specfem::program::Context, so they would
// otherwise skip the Context's device selection and verification and silently
// run on AdaptiveCpp's OpenMP host device.

namespace specfem::testing_impl {

/**
 * @brief GoogleTest listener that selects the Metal GPU for a test process
 * and fails the process if Kokkos runs anywhere else.
 *
 * Selection (at program start, before any test or environment initializes
 * Kokkos) reuses specfem::program_impl::select_default_device(): it finds the
 * Metal GPU by backend, independent of device enumeration order, and leaves
 * an explicit KOKKOS_DEVICE_ID / KOKKOS_VISIBLE_DEVICES /
 * KOKKOS_MAP_DEVICE_ID_BY untouched.
 *
 * Verification reuses specfem::program_impl::check_execution_device(), which
 * honours SPECFEM_METAL_ALLOW_HOST_DEVICE for diagnostic host-device runs. It
 * runs once per process, as soon as Kokkos is observed initialized: after
 * global environment set-up, around every test, and from a Kokkos finalize
 * hook for fixtures that initialize and finalize Kokkos inside one test. A
 * wrong device terminates the process with a nonzero exit status, so CTest
 * reports the test as failed.
 *
 * GoogleTest does not forward events to listeners in death-test child
 * processes, so the listener never acts there; children may select devices
 * deliberately (DeviceSelection.RejectsHostDevice). Death tests are switched
 * to the "threadsafe" (re-executing) style, because device selection starts
 * the AdaptiveCpp runtime in the parent and a fork()ed child must not use it.
 */
class MetalDeviceCheck : public ::testing::EmptyTestEventListener {
public:
  void OnTestProgramStart(const ::testing::UnitTest &) override {
    // The child of a threadsafe death test takes its style from the
    // environment (it receives no listener events); with the default "fast"
    // style it would fork again instead of running the death statement.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ::setenv("GTEST_DEATH_TEST_STYLE", "threadsafe", 1);
    try {
      specfem::program_impl::select_default_device();
    } catch (const std::exception &e) {
      fail(e.what());
    }
    Kokkos::push_finalize_hook([] { verify(); });
  }

  void OnEnvironmentsSetUpEnd(const ::testing::UnitTest &) override {
    verify_if_initialized();
  }

  void OnTestStart(const ::testing::TestInfo &) override {
    verify_if_initialized();
  }

  void OnTestEnd(const ::testing::TestInfo &) override {
    verify_if_initialized();
  }

private:
  static void verify_if_initialized() {
    if (Kokkos::is_initialized()) {
      verify();
    }
  }

  static void verify() {
    // Kokkos cannot be re-initialized after finalize, so the device a process
    // first initializes Kokkos on is the only one it can use.
    static bool verified = false;
    if (verified) {
      return;
    }
    try {
      const std::string device =
          specfem::program_impl::check_execution_device();
      verified = true;
      std::printf("[specfem device check] %s\n", device.c_str());
      std::fflush(stdout);
    } catch (const std::exception &e) {
      fail(e.what());
    }
  }

  [[noreturn]] static void fail(const char *what) {
    std::fprintf(stderr, "[specfem device check] FAILED: %s\n", what);
    std::fflush(stderr);
    // No unwinding: this may run inside a Kokkos finalize hook (noexcept).
    std::_Exit(EXIT_FAILURE);
  }
};

/**
 * @brief Registers MetalDeviceCheck before main() runs.
 *
 * Registration through a static object keeps every test's main() unchanged.
 * UnitTest::GetInstance() is a function-local static, so this is safe during
 * static initialization; GoogleTest takes ownership of the listener.
 */
struct MetalDeviceCheckRegistration {
  MetalDeviceCheckRegistration() {
    ::testing::UnitTest::GetInstance()->listeners().Append(
        new MetalDeviceCheck);
  }
};

const MetalDeviceCheckRegistration metal_device_check_registration{};

} // namespace specfem::testing_impl
