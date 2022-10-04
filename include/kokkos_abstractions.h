#ifndef KOKKOS_ABSTRACTION_H
#define KOKKOS_ABSTRACTION_H

#include "../include/config.h"
#include <Kokkos_Core.hpp>

namespace specfem {

using HostMemSpace = Kokkos::HostSpace;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using DevExecSpace = Kokkos::DefaultExecutionSpace;

// Dafault layout
using LayoutWrapper = Kokkos::LayoutRight;

// Default iterate policy for MDRange
// using IterateWrapper = Kokkos::Iterate::Right;

// scratch memory spaces
using HostScratchSpace = HostExecSpace::scratch_memory_space;
using DevScratchSpace = DevExecSpace::scratch_memory_space;

// Device views
template <typename T, typename L = LayoutWrapper>
using DeviceView1d = Kokkos::View<T *, L, DevMemSpace>;
template <typename T, typename L = LayoutWrapper>
using DeviceView2d = Kokkos::View<T **, L, DevMemSpace>;

// Host views
template <typename T, typename L = LayoutWrapper>
using HostView1d = Kokkos::View<T *, L, HostMemSpace>;
template <typename T, typename L = LayoutWrapper>
using HostView2d = Kokkos::View<T **, L, HostMemSpace>;
template <typename T, typename L = LayoutWrapper>
using HostView3d = Kokkos::View<T ***, L, HostMemSpace>;
template <typename T, typename L = LayoutWrapper>
using HostView4d = Kokkos::View<T ****, L, HostMemSpace>;

// Host Mirrors
template <typename T, typename L = LayoutWrapper>
using HostMirror1d = typename DeviceView1d<T, L>::HostMirror;
template <typename T, typename L = LayoutWrapper>
using HostMirror2d = typename DeviceView2d<T, L>::HostMirror;

// Scratch Views
template <typename T, typename L = LayoutWrapper>
using HostScratchView1d =
    Kokkos::View<T *, L, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
template <typename T, typename L = LayoutWrapper>
using HostScratchView2d =
    Kokkos::View<T **, L, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
template <typename T, typename L = LayoutWrapper>
using HostScratchView3d =
    Kokkos::View<T ***, L, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

template <typename T, typename L = LayoutWrapper>
using DevScratchView1d = Kokkos::View<T *, L, DevScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
template <typename T, typename L = LayoutWrapper>
using DevScratchView2d = Kokkos::View<T **, L, DevScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
template <typename T, typename L = LayoutWrapper>
using DevScratchView3d = Kokkos::View<T ***, L, DevScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

// Loop Strategies
// Range policy strategies

template <int T, Kokkos::Iterate IteratePolicy = Kokkos::Iterate::Right>
using HostMDrange =
    Kokkos::MDRangePolicy<HostExecSpace, Kokkos::Rank<T, IteratePolicy> >;

// Team Policy Strategy

// TODO add launch bounds

using HostTeam = Kokkos::TeamPolicy<HostExecSpace>;
using DevTeam = Kokkos::TeamPolicy<DevExecSpace>;

} // namespace specfem

#endif
