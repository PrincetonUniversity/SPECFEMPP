#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem::chunk_element {

/// @brief Primary template — undefined; use 1-, 2-, or 3-field specializations.
template <typename... Fs> struct FieldPack;

// ---------------------------------------------------------------------------
// 1-field specialization
// ---------------------------------------------------------------------------

/**
 * @brief Single-field pack (accessed as .f).
 *
 * @tparam F Chunk element field type (e.g., chunk_element::displacement<...>)
 */
template <typename F> struct FieldPack<F> {
  static constexpr std::size_t size = 1;
  F f;

  KOKKOS_FUNCTION FieldPack() = default;
  KOKKOS_FUNCTION FieldPack(const FieldPack &) = default;
  KOKKOS_FUNCTION FieldPack &operator=(const FieldPack &) = default;

  /// @brief Construct directly from an existing field value (e.g., in tests).
  KOKKOS_FUNCTION explicit FieldPack(const F &f_) : f(f_) {}

  /// @brief Construct from a Kokkos scratch memory space.
  /// SFINAE guard prevents this overload from being selected when ScratchSpace
  /// is F itself (which would cause ambiguity with the value constructor).
  template <typename S,
            std::enable_if_t<!std::is_same_v<std::decay_t<S>, F>, int> = 0>
  KOKKOS_FUNCTION FieldPack(const S &scratch) : f(scratch) {}

  /// @brief Shared memory size required for the field.
  static std::size_t shmem_size() { return F::shmem_size(); }
};

// ---------------------------------------------------------------------------
// 2-field specialization
// ---------------------------------------------------------------------------

/**
 * @brief Two-field pack (accessed as .f and .g).
 *
 * Member initialization order (f then g) follows declaration order —
 * a C++ standard guarantee — so each field advances the shared Kokkos scratch
 * pointer sequentially without overlap.
 *
 * @tparam F First chunk element field type
 * @tparam G Second chunk element field type
 */
template <typename F, typename G> struct FieldPack<F, G> {
  static constexpr std::size_t size = 2;
  F f;
  G g;

  KOKKOS_FUNCTION FieldPack() = default;
  KOKKOS_FUNCTION FieldPack(const FieldPack &) = default;
  KOKKOS_FUNCTION FieldPack &operator=(const FieldPack &) = default;

  KOKKOS_FUNCTION FieldPack(const F &f_, const G &g_) : f(f_), g(g_) {}

  template <typename S,
            std::enable_if_t<!std::is_same_v<std::decay_t<S>, F> &&
                                 !std::is_same_v<std::decay_t<S>, G>,
                             int> = 0>
  KOKKOS_FUNCTION FieldPack(const S &scratch) : f(scratch), g(scratch) {}

  static std::size_t shmem_size() { return F::shmem_size() + G::shmem_size(); }
};

// ---------------------------------------------------------------------------
// 3-field specialization
// ---------------------------------------------------------------------------

/**
 * @brief Three-field pack (accessed as .f, .g, and .h).
 *
 * @tparam F First chunk element field type
 * @tparam G Second chunk element field type
 * @tparam H Third chunk element field type
 */
template <typename F, typename G, typename H> struct FieldPack<F, G, H> {
  static constexpr std::size_t size = 3;
  F f;
  G g;
  H h;

  KOKKOS_FUNCTION FieldPack() = default;
  KOKKOS_FUNCTION FieldPack(const FieldPack &) = default;
  KOKKOS_FUNCTION FieldPack &operator=(const FieldPack &) = default;

  KOKKOS_FUNCTION FieldPack(const F &f_, const G &g_, const H &h_)
      : f(f_), g(g_), h(h_) {}

  template <typename S,
            std::enable_if_t<!std::is_same_v<std::decay_t<S>, F> &&
                                 !std::is_same_v<std::decay_t<S>, G> &&
                                 !std::is_same_v<std::decay_t<S>, H>,
                             int> = 0>
  KOKKOS_FUNCTION FieldPack(const S &scratch)
      : f(scratch), g(scratch), h(scratch) {}

  static std::size_t shmem_size() {
    return F::shmem_size() + G::shmem_size() + H::shmem_size();
  }
};

} // namespace specfem::chunk_element
