#include "specfem/program/device_selection.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  // Kokkos is initialized before RUN_ALL_TESTS, so select the device here
  // (AdaptiveCpp/Metal builds; no-op elsewhere).
  specfem::program_impl::select_default_device();
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
