#ifndef KOKKOS_ABSTRACTION_H
#define KOKKOS_ABSTRACTION_H

#include "../include/config.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

namespace specfem {

/** @name Execution Spaces
 */
///@{
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using DevExecSpace = Kokkos::DefaultExecutionSpace;
///@}

/** @name Memory Spaces
 */
///@{
using HostMemSpace = Kokkos::HostSpace;
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
///@}

/** @name View Layout
 * @note The default view layoput is set to Kokkos::LayoutRight format for all
 * views. In most cases views are accessed inside team-policies (Host or
 * Device). Which means a LayoutRight data layout results in caching as well as
 * coalescing.
 */
///@{
using LayoutWrapper = Kokkos::LayoutRight;
///@}

/** @name Scratch Memory Spaces
 */
///@{
using HostScratchSpace = HostExecSpace::scratch_memory_space;
using DevScratchSpace = DevExecSpace::scratch_memory_space;
///@}

/** @name Device views
 */
///@{
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DeviceView1d = Kokkos::View<T *, L, DevMemSpace>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DeviceView2d = Kokkos::View<T **, L, DevMemSpace>;
///@}

/** @name Host views
 */
///@{

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostView1d = Kokkos::View<T *, L, HostMemSpace>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostView2d = Kokkos::View<T **, L, HostMemSpace>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostView3d = Kokkos::View<T ***, L, HostMemSpace>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostView4d = Kokkos::View<T ****, L, HostMemSpace>;
///@}

/** @name Host Scatter Views
 * Scatter views are used when an atomic update is needed on a view.
 * Scattered access optimizes for atomic algorithm used when updating a view.
 * For more details check Kokkos scatter view tutorials.
 */

///@{
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostScatterView1d =
    Kokkos::Experimental::ScatterView<T *, L, HostExecSpace>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostScatterView2d =
    Kokkos::Experimental::ScatterView<T **, L, HostExecSpace>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostScatterView3d =
    Kokkos::Experimental::ScatterView<T ***, L, HostExecSpace>;
///}

/** @name Device Scatter Views
 * Scatter views are used when an atomic update is needed on a view.
 * Scattered access optimizes for atomic algorithm used when updating a view.
 * For more details check Kokkos scatter view tutorials.
 */

///@{
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DeviceScatterView1d =
    Kokkos::Experimental::ScatterView<T *, L, DevExecSpace>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DeviceScatterView2d =
    Kokkos::Experimental::ScatterView<T **, L, DevExecSpace>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DeviceScatterView3d =
    Kokkos::Experimental::ScatterView<T ***, L, DevExecSpace>;
///}

/** @name Host Mirrors
 *
 * Use host mirrors of device views to sync data between host and device
 */
///@{

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostMirror1d = typename DeviceView1d<T, L>::HostMirror;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostMirror2d = typename DeviceView2d<T, L>::HostMirror;
///@}

// Scratch Views
/** @name Scratch views
 * Scratch views are generally used to allocate data in kokkos scratch memory
 * spaces. Optimal use of scrach memory spaces can optimize bandwidth of the
 * kernel
 */
///@{
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostScratchView1d =
    Kokkos::View<T *, L, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostScratchView2d =
    Kokkos::View<T **, L, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using HostScratchView3d =
    Kokkos::View<T ***, L, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DevScratchView1d = Kokkos::View<T *, L, DevScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DevScratchView2d = Kokkos::View<T **, L, DevScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DevScratchView3d = Kokkos::View<T ***, L, DevScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
///@}

// Loop Strategies
// Range policy strategies

/** @name Range policies
 *
 * TODO : have an example code here on how to use HostMDrange policy
 *
 */
///@{
/**
 * @brief Host multi-dimensional range policy
 *
 * TODO : have an example code here on how to use HostMDrange policy
 *
 * @tparam T Number of view dimensions to collapse inside the loop
 * @tparam IteratePolicy
 */
template <int T, Kokkos::Iterate IteratePolicy = Kokkos::Iterate::Right>
using HostMDrange =
    Kokkos::MDRangePolicy<HostExecSpace, Kokkos::Rank<T, IteratePolicy> >;

/**
 * @brief Host range policy
 *
 */
using HostRange = Kokkos::RangePolicy<HostExecSpace>;

/**
 * @brief Device multi-dimensional range policy
 *
 * TODO : have an example code here on how to use DeviceMDrange policy
 *
 * @tparam T Number of view dimensions to collapse inside the loop
 * @tparam IteratePolicy
 */
template <int T, Kokkos::Iterate IteratePolicy = Kokkos::Iterate::Right>
using DevMDrange =
    Kokkos::MDRangePolicy<DevExecSpace, Kokkos::Rank<T, IteratePolicy> >;

/**
 * @brief Device range policy
 *
 */
using DevRange = Kokkos::RangePolicy<DevExecSpace>;
///@}

// Team Policy Strategy

// TODO add launch bounds

/** @name Team policies
 */
///@{
using HostTeam = Kokkos::TeamPolicy<HostExecSpace>;
using DevTeam = Kokkos::TeamPolicy<DevExecSpace>;
///@}

} // namespace specfem

#endif
