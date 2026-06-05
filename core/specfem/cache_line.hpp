#pragma once

#include <Kokkos_Core.hpp>

#ifdef _MSC_VER
#include <intrin.h> // _MM_HINT_* constants
#endif

namespace specfem {
namespace cache_line {

/**
 * @brief Non-temporal prefetch hint: bypass L2/L3 to L1 only.
 * Useful for data used once or streamed sequentially.
 */
struct Nta {
  // Non-temporal — stream to L1 bypassing L2/L3
#ifdef _MSC_VER
  static constexpr auto kHint = _MM_HINT_NTA;
#else
  static constexpr auto kHint = 0; // __builtin_prefetch: no temporal locality
#endif
};

/**
 * @brief Low temporal locality prefetch hint: L3/last-level cache only.
 * For data reused infrequently across large strides.
 */
struct Low {
  // L3 / last-level cache only
#ifdef _MSC_VER
  static constexpr auto kHint = _MM_HINT_T2;
#else
  static constexpr auto kHint = 1; // __builtin_prefetch: low locality
#endif
};

/**
 * @brief Moderate temporal locality prefetch hint: L2 and above.
 * For data with medium reuse distance.
 */
struct Moderate {
  // L2 and above
#ifdef _MSC_VER
  static constexpr auto kHint = _MM_HINT_T1;
#else
  static constexpr auto kHint = 2; // __builtin_prefetch: moderate locality
#endif
};

/**
 * @brief High temporal locality prefetch hint: L1, L2, L3 (default).
 * For data reused within short windows.
 */
struct High {
  // L1, L2, L3 (default)
#ifdef _MSC_VER
  static constexpr auto kHint = _MM_HINT_T0;
#else
  static constexpr auto kHint = 3; // __builtin_prefetch: full locality
#endif
};

/**
 * @brief Issue a software prefetch hint for a pointer.
 *
 * Brings data into cache with the specified locality hint.
 * No-op on CUDA. Cross-platform hint mapping for x86 and ARM.
 *
 * @tparam LocalityHint Hint type (High, Moderate, Low, Nta)
 * @tparam T Pointer type
 * @param ptr Address to prefetch
 * @param hint Cache locality hint (default: High)
 *
 * @code
 * auto *data = get_data();
 * specfem::cache_line::prefetch<specfem::cache_line::Moderate>(data);
 * @endcode
 */
template <typename LocalityHint = High, typename T>
KOKKOS_FORCEINLINE_FUNCTION void prefetch(const T *ptr, LocalityHint = {}) {
#if defined(KOKKOS_ENABLE_CUDA)
  (void)ptr;
#elif defined(_MSC_VER)
  _mm_prefetch(reinterpret_cast<const char *>(ptr), LocalityHint::kHint);
#else
  __builtin_prefetch(ptr, 0, LocalityHint::kHint);
#endif
}

} // namespace cache_line
} // namespace specfem
