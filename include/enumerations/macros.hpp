#pragma once
#include <Kokkos_Core.hpp>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define KOKKOS_ABORT_WITH_LOCATION(message)                                    \
  Kokkos::abort(__FILE__ ":" TOSTRING(__LINE__) " - " message);
