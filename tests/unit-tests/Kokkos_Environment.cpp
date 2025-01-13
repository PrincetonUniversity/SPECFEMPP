#include "Kokkos_Environment.hpp"
#include <Kokkos_Core.hpp>

void KokkosEnvironment::SetUp() {
  char **argv = nullptr;
  int argc = 0;
  Kokkos::initialize(argc, argv);
}

void KokkosEnvironment::TearDown() { Kokkos::finalize(); }
