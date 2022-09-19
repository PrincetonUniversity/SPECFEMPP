#include "Kokkos_Environment.hpp"
#include <Kokkos_Core.hpp>

void KokkosEnvironment::SetUp() {
  char **argv;
  int argc = 0;
  Kokkos::initialize(argc, argv);
}

void KokkosEnvironment::TearDown() { Kokkos::finalize(); }
