#pragma once

namespace specfem {
namespace utilities {

/**
 * @brief Always-false template variable helper for static assertions.
 *
 * Used in if-constexpr branches and static_assert statements to generate
 * compile-time errors only when a specific template instantiation occurs.
 * This is more robust than using false directly, which could trigger
 * warnings or be optimized incorrectly.
 *
 * @tparam T Template parameter pack (values are ignored)
 *
 * @code
 * template<typename T>
 * void process() {
 *   if constexpr (std::is_same_v<T, int>) {
 *     // handle int
 *   } else {
 *     static_assert(always_false<T>, "Unsupported type");
 *   }
 * }
 * @endcode
 */
template <auto... T> constexpr bool always_false = false;

} // namespace utilities
} // namespace specfem
