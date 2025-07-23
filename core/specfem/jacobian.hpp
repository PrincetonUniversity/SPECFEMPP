#pragma once
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"

namespace specfem::jacobian {
using HostCoord =
    Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::HostSpace>;
using HostScratchCoord =
    Kokkos::View<type_real **, Kokkos::LayoutLeft,
                 Kokkos::DefaultHostExecutionSpace::scratch_memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
} // namespace specfem::jacobian

#include "jacobian/dim2/jacobian.hpp"
#include "jacobian/dim3/jacobian.hpp"
