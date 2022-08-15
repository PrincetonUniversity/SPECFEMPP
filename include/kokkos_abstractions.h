#ifndef KOKKOS_ABSTRACTION_H
#define KOKKOS_ABSTRACTION_H

#include "../include/config.h"
#include <Kokkos_Core.hpp>

namespace specfem {

using HostMemSpace = Kokkos::HostSpace;
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using DevExecSpace = Kokkos::DefaultExecutionSpace;

// Device views
template <typename T> using DeviceView1d = Kokkos::View<T *, DevMemSpace>;
template <typename T> using DeviceView2d = Kokkos::View<T **, DevMemSpace>;

// Host views
template <typename T> using HostView1d = Kokkos::View<T *, HostMemSpace>;
template <typename T> using HostView2d = Kokkos::View<T **, HostMemSpace>;

// Host Mirrors
template <typename T> using HostMirror1d = typename DeviceView1d<T>::HostMirror;
template <typename T> using HostMirror2d = typename DeviceView2d<T>::HostMirror;

} // namespace specfem

#endif
