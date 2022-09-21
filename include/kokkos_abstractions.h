#ifndef KOKKOS_ABSTRACTION_H
#define KOKKOS_ABSTRACTION_H

#include "../include/config.h"
#include <Kokkos_Core.hpp>

namespace specfem {

using HostMemSpace = Kokkos::HostSpace;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using DevExecSpace = Kokkos::DefaultExecutionSpace;

// scratch memory spaces
using HostScratchSpace = HostExecSpace::scratch_memory_space;
using DevScratchSpace = DevExecSpace::scratch_memory_space;

// Device views
template <typename T> using DeviceView1d = Kokkos::View<T *, DevMemSpace>;
template <typename T> using DeviceView2d = Kokkos::View<T **, DevMemSpace>;

// Host views
template <typename T> using HostView1d = Kokkos::View<T *, HostMemSpace>;
template <typename T> using HostView2d = Kokkos::View<T **, HostMemSpace>;
template <typename T> using HostView3d = Kokkos::View<T ***, HostMemSpace>;

// Host Mirrors
template <typename T> using HostMirror1d = typename DeviceView1d<T>::HostMirror;
template <typename T> using HostMirror2d = typename DeviceView2d<T>::HostMirror;

// Scratch Views
template <typename T>
using HostScratchView1d =
    Kokkos::View<T *, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
template <typename T>
using HostScratchView2d =
    Kokkos::View<T **, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
template <typename T>
using HostScratchView3d =
    Kokkos::View<T ***, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

template <typename T>
using DevScratchView1d = Kokkos::View<T *, DevScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
template <typename T>
using DevScratchView2d = Kokkos::View<T **, DevScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
template <typename T>
using DevScratchView3d = Kokkos::View<T ***, DevScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

// Loop Strategies
// Range policy strategies

template <int T>
using HostMDrange = Kokkos::MDRangePolicy<HostExecSpace, Kokkos::Rank<T> >;

// Team Policy Strategy

// TODO add launch bounds

using HostTeam = Kokkos::TeamPolicy<HostExecSpace>;
using DevTeam = Kokkos::TeamPolicy<DevExecSpace>;

} // namespace specfem

#endif
