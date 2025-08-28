#ifndef KOKKOS_ABSTRACTION_H
#define KOKKOS_ABSTRACTION_H

#include "../include/specfem_setup.hpp"
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

/**
 * @namespace Defines views and execution policies used throughout the SPECFEM
 * project
 *
 */
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

/** @name Static Device views
 */
///@{
/**
 * @tparam T view datatype
 * @tparam N view size
 * @tparam L view layout - default layout is LayoutRight
 *  @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 *
 * @code ...
 *    StridedCacheAlignedView1d = StaticDeviceView1d<double, 100,
 * Kokkos::MemoryTraits<Kokkos::Aligned>>(...)
 * @endcode
 */
template <typename T, int N, typename L = LayoutWrapper, typename... Args>
using StaticDeviceView1d = Kokkos::View<T[N], L, DevMemSpace, Args...>;

/**
 * @tparam T view datatype
 * @tparam N view size
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, int N, typename L = LayoutWrapper, typename... Args>
using StaticDeviceView2d = Kokkos::View<T[N][N], L, DevMemSpace, Args...>;

/**
 * @tparam T view datatype
 * @tparam N view size
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, int N, typename L = LayoutWrapper, typename... Args>
using StaticDeviceView3d = Kokkos::View<T[N][N][N], L, DevMemSpace, Args...>;
///@}

/** @name Static Host views
 */
///@{

/**
 * @tparam T view datatype
 * @tparam N view size
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, int N, typename L = LayoutWrapper, typename... Args>
using StaticHostView1d = Kokkos::View<T[N], L, HostMemSpace, Args...>;

/**
 * @tparam T view datatype
 * @tparam N view size
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, int N, typename L = LayoutWrapper, typename... Args>
using StaticHostView2d = Kokkos::View<T[N][N], L, HostMemSpace, Args...>;

/**
 * @tparam T view datatype
 * @tparam N view size
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, int N, typename L = LayoutWrapper, typename... Args>
using StaticHostView3d = Kokkos::View<T[N][N][N], L, HostMemSpace, Args...>;
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
/**
 * @brief 6d device view
 *
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using DeviceView6d = Kokkos::View<T ******, L, DevMemSpace, Args...>;
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
/**
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostView5d = Kokkos::View<T *****, L, HostMemSpace, Args...>;
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
/**
 * @brief Host mirror of 6d device view
 *
 * @tparam T view datatype
 * @tparam L view layout - default layout is LayoutRight
 * @tparam Args - Args can be used to customize your views. These are passed
 * directly to Kokkos::Views objects
 */
template <typename T, typename L = LayoutWrapper, typename... Args>
using HostMirror6d = typename DeviceView6d<T, L, Args...>::HostMirror;
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
using simd_type = Kokkos::Experimental::basic_simd<T, simd_abi>;

/**
 * @name Custom reductions for Kokkos TeamThreadRange policies.
 *
 * These reductions are used in Kokkos nested policies. Kokkos required
 * nested policy reductions to be reduced into scalar types. Use of these
 * reduction policies would be appropriate when reductions need to be done into
 * arrays - for example when computing seismograms (check
 * domain.tpp::compute_seismograms() for examples on how to use this)
 *
 */
///@{
/**
 * Sum reduction
 *
 * @tparam T Scalar Array types for reductions. Can be either <dim2 or
 * dim3>::<array_type or scalar_type>
 */
template <typename T, class Space = DevMemSpace> class Sum {
public:
  // Required typedefs
  /**
   * @brief Check Kokkos custom reducers for more details
   * (https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Custom-Reductions-Custom-Reducers.html)
   *
   */
  typedef T value_type; ///< Value type of reduction
  typedef Sum reducer;  ///< Required typedef for reduction
  typedef Kokkos::View<value_type *, Space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      result_view_type; ///< Required typedef for reduction

  /**
   * @brief Constructor
   *
   * @param value - reference to value to be reduced into
   */
  KOKKOS_INLINE_FUNCTION Sum(value_type &value) : value(value) {}

  /**
   * @brief init operator to initialize value to be reduced
   *
   * @param update value to be reduced into
   * @return KOKKOS_INLINE_FUNCTION
   */
  KOKKOS_INLINE_FUNCTION void init(value_type &update) const { update.init(); }

  /**
   * @brief join operator to join values from different threads
   *
   * @param update value to be reduced into
   * @param source value to be reduced from
   */
  KOKKOS_INLINE_FUNCTION void join(value_type &update,
                                   const value_type &source) const {
    update += source;
  }

  /**
   * @brief reference operator to return reference to value to be reduced into
   *
   */
  KOKKOS_INLINE_FUNCTION value_type &reference() const { return value; }

  /**
   * @brief view operator to return view of value to be reduced into
   *
   */
  KOKKOS_INLINE_FUNCTION result_view_type view() const {
    return result_view_type(&value, 1);
  }

  /**
   * @brief references_scalar operator to return true if value to be reduced is
   * a scalar type
   *
   */
  KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }

private:
  value_type &value; ///< Reference to value to be reduced into
};
///@}

} // namespace kokkos
} // namespace specfem

#endif
