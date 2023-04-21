#ifndef KOKKOS_ABSTRACTION_H
#define KOKKOS_ABSTRACTION_H

#include "../include/config.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <Kokkos_ScatterView.hpp>

namespace specfem {

namespace sync {
enum kind {
  HostToDevice, ///< Sync data from host to device
  DeviceToHost  ///< Sync data from device to host
};
} // namespace sync

namespace kokkos {
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
using LayoutStride = Kokkos::LayoutStride;
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
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 *
 * @code ...
 *      StridedCacheAlignedView1d = DeviceView1d<double, LayoutStride,
 * Kokkos::MemoryTraits<Kokkos::Aligned>>(...)
 * @endcode
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using DeviceView1d = Kokkos::View<T *, L, DevMemSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using DeviceView2d = Kokkos::View<T **, L, DevMemSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using DeviceView3d = Kokkos::View<T ***, L, DevMemSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using DeviceView4d = Kokkos::View<T ****, L, DevMemSpace, Args...>;
///@}
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using DeviceView5d = Kokkos::View<T *****, L, DevMemSpace, Args...>;
///@}

/** @name Host views
 */
///@{

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostView1d = Kokkos::View<T *, L, HostMemSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostView2d = Kokkos::View<T **, L, HostMemSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostView3d = Kokkos::View<T ***, L, HostMemSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostView4d = Kokkos::View<T ****, L, HostMemSpace, Args...>;
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
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostScatterView1d =
    Kokkos::Experimental::ScatterView<T *, L, HostExecSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostScatterView2d =
    Kokkos::Experimental::ScatterView<T **, L, HostExecSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostScatterView3d =
    Kokkos::Experimental::ScatterView<T ***, L, HostExecSpace, Args...>;
///@}

/** @name Device Scatter Views
 * Scatter views are used when an atomic update is needed on a view.
 * Scattered access optimizes for atomic algorithm used when updating a view.
 * For more details check Kokkos scatter view tutorials.
 */

///@{
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using DeviceScatterView1d =
    Kokkos::Experimental::ScatterView<T *, L, DevExecSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using DeviceScatterView2d =
    Kokkos::Experimental::ScatterView<T **, L, DevExecSpace, Args...>;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using DeviceScatterView3d =
    Kokkos::Experimental::ScatterView<T ***, L, DevExecSpace, Args...>;
///@}

/** @name Host Mirrors
 *
 * Use host mirrors of device views to sync data between host and device
 */
///@{

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostMirror1d = typename DeviceView1d<T, L, Args...>::HostMirror;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostMirror2d = typename DeviceView2d<T, L, Args...>::HostMirror;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostMirror3d = typename DeviceView3d<T, L, Args...>::HostMirror;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostMirror4d = typename DeviceView4d<T, L, Args...>::HostMirror;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostMirror5d = typename DeviceView5d<T, L, Args...>::HostMirror;
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
using DeviceScratchView1d =
    Kokkos::View<T *, L, DevScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DeviceScratchView2d =
    Kokkos::View<T **, L, DevScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 */
template <typename T, typename L = LayoutWrapper>
using DeviceScratchView3d =
    Kokkos::View<T ***, L, DevScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
///@}

/** @name Static Scratch Views
 * Static scratch views are best used when scratch view dimensions are known at
 * compile time. Compile time definitions of scratch views and loop structure
 * can dramatically improve code performance.
 */

///@{

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam N length of view
 */
template <typename T, int N1, typename L = LayoutWrapper>
using StaticHostScratchView1d =
    Kokkos::View<T[N1], L, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam N length of view
 */
template <typename T, int N1, int N2, typename L = LayoutWrapper>
using StaticHostScratchView2d =
    Kokkos::View<T[N1][N2], L, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam N length of view
 */
template <typename T, int N1, int N2, int N3, typename L = LayoutWrapper>
using StaticHostScratchView3d =
    Kokkos::View<T[N1][N2][N3], L, HostScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam N length of view
 */
template <typename T, int N1, typename L = LayoutWrapper>
using StaticDeviceScratchView1d =
    Kokkos::View<T[N1], L, DevScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam N length of view
 */
template <typename T, int N1, int N2, typename L = LayoutWrapper>
using StaticDeviceScratchView2d =
    Kokkos::View<T[N1][N2], L, DevScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam N length of view
 */
template <typename T, int N1, int N2, int N3, typename L = LayoutWrapper>
using StaticDeviceScratchView3d =
    Kokkos::View<T[N1][N2][N3], L, DevScratchSpace,
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
using DeviceMDrange =
    Kokkos::MDRangePolicy<DevExecSpace, Kokkos::Rank<T, IteratePolicy> >;

/**
 * @brief Device range policy
 *
 */
using DeviceRange = Kokkos::RangePolicy<DevExecSpace>;
///@}

// Team Policy Strategy

// TODO add launch bounds

/** @name Team policies
 */
///@{
using HostTeam = Kokkos::TeamPolicy<HostExecSpace>;
using DeviceTeam = Kokkos::TeamPolicy<DevExecSpace>;
///@}

/**
 * @brief Enable SIMD intrinsics using SIMD variables.
 *
 * @tparam T type T of variable
 * @tparam simd_abi simd_abi value can either be native or scalar. These values
 * determine if the Kokkos enables SIMD vectorization or reverts to scalar
 * operator implementations
 *
 * @note Currently there is a bug in Kokkos SIMD implemtation when compiling
 * with GCC or clang. If using SIMD vectorization pass the @code -fpermissive
 * @endcode flag to CXX compiler
 */
template <typename T = type_real,
          typename simd_abi = Kokkos::Experimental::simd_abi::scalar>
using simd_type = Kokkos::Experimental::simd<T, simd_abi>;

} // namespace kokkos
} // namespace specfem

#endif
