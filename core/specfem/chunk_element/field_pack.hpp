#pragma once

#include "specfem/point/gradient_field_pack.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem::chunk_element {

/**
 * @brief Named holder for a displacement chunk field (accessed as .u)
 *
 * Carries the nested `gradient_holder<V>` alias that maps this holder to
 * `specfem::point::holds_du<V>` for use in FieldPack gradient overloads.
 *
 * The constructor takes a Kokkos scratch memory space; the underlying field
 * allocates its storage from that space. When multiple holders are composed
 * via FieldPack, they are initialized in declaration order (C++ standard
 * guarantee), so each holder advances the shared scratch pointer sequentially
 * without overlap.
 *
 * @tparam F Chunk element field type (e.g., chunk_element::displacement<...>)
 */
template <typename F> struct holds_u {
  F u;

  /// Maps this holder to its gradient counterpart holds_du<V>
  template <typename V> using gradient_holder = specfem::point::holds_du<V>;

  /// @brief Access the underlying field (used by gradient overloads)
  KOKKOS_FUNCTION F &get() { return u; }
  KOKKOS_FUNCTION const F &get() const { return u; }

  KOKKOS_FUNCTION holds_u() = default;
  KOKKOS_FUNCTION holds_u(const holds_u &) = default;
  KOKKOS_FUNCTION holds_u &operator=(const holds_u &) = default;

  /// @brief Construct directly from an existing field value (e.g., in tests)
  KOKKOS_FUNCTION explicit holds_u(const F &f) : u(f) {}

  /// @brief Construct from a Kokkos scratch memory space.
  /// SFINAE guard prevents this overload from being selected when ScratchSpace
  /// is F itself (which would cause ambiguity with the value constructor).
  template <
      typename ScratchSpace,
      std::enable_if_t<!std::is_same_v<std::decay_t<ScratchSpace>, F>, int> = 0>
  KOKKOS_FUNCTION holds_u(const ScratchSpace &scratch) : u(scratch) {}

  /// @brief Shared memory size required for the field
  static std::size_t shmem_size() { return F::shmem_size(); }
};

/**
 * @brief Named holder for a velocity chunk field (accessed as .v)
 *
 * Carries the nested `gradient_holder<V>` alias that maps this holder to
 * `specfem::point::holds_dv<V>`.
 *
 * @tparam F Chunk element field type (e.g., chunk_element::velocity<...>)
 */
template <typename F> struct holds_v {
  F v;

  template <typename V> using gradient_holder = specfem::point::holds_dv<V>;

  KOKKOS_FUNCTION F &get() { return v; }
  KOKKOS_FUNCTION const F &get() const { return v; }

  KOKKOS_FUNCTION holds_v() = default;
  KOKKOS_FUNCTION holds_v(const holds_v &) = default;
  KOKKOS_FUNCTION holds_v &operator=(const holds_v &) = default;

  KOKKOS_FUNCTION explicit holds_v(const F &f) : v(f) {}

  template <
      typename ScratchSpace,
      std::enable_if_t<!std::is_same_v<std::decay_t<ScratchSpace>, F>, int> = 0>
  KOKKOS_FUNCTION holds_v(const ScratchSpace &scratch) : v(scratch) {}

  static std::size_t shmem_size() { return F::shmem_size(); }
};

/**
 * @brief Named holder for an acceleration chunk field (accessed as .a)
 *
 * Carries the nested `gradient_holder<V>` alias that maps this holder to
 * `specfem::point::holds_da<V>`.
 *
 * @tparam F Chunk element field type (e.g., chunk_element::acceleration<...>)
 */
template <typename F> struct holds_a {
  F a;

  template <typename V> using gradient_holder = specfem::point::holds_da<V>;

  KOKKOS_FUNCTION F &get() { return a; }
  KOKKOS_FUNCTION const F &get() const { return a; }

  KOKKOS_FUNCTION holds_a() = default;
  KOKKOS_FUNCTION holds_a(const holds_a &) = default;
  KOKKOS_FUNCTION holds_a &operator=(const holds_a &) = default;

  KOKKOS_FUNCTION explicit holds_a(const F &f) : a(f) {}

  template <
      typename ScratchSpace,
      std::enable_if_t<!std::is_same_v<std::decay_t<ScratchSpace>, F>, int> = 0>
  KOKKOS_FUNCTION holds_a(const ScratchSpace &scratch) : a(scratch) {}

  static std::size_t shmem_size() { return F::shmem_size(); }
};

/**
 * @brief Variadic named-holder pack for chunk element fields.
 *
 * Bundles multiple named field holders (holds_u, holds_v, holds_a) via
 * multiple inheritance. Base classes are initialized in declaration/parameter
 * order (C++ standard guarantee), so scratch memory is allocated sequentially
 * without overlap even when multiple fields share the same
 * `team.team_scratch(0)`.
 *
 * Example:
 * @code
 * using Pack =
 *     FieldPack<holds_u<DisplacementType>, holds_v<VelocityType>>;
 * Pack pack(team.team_scratch(0));
 * pack.u  // displacement field
 * pack.v  // velocity field
 * @endcode
 *
 * Usage in compute_stiffness_interaction:
 * @code
 * // scratch_size uses FieldPack::shmem_size()
 * int scratch_size = ChunkFieldPackType::shmem_size() + ...;
 * ChunkFieldPackType field_pack(team.team_scratch(0));
 * specfem::assembly::load_on_device(chunk_index, field, field_pack.u);
 * if constexpr (needs_velocity) {
 *   specfem::assembly::load_on_device(chunk_index, field, field_pack.v);
 * }
 * @endcode
 *
 * @tparam Holders Variadic list of named holder types
 */
template <typename... Holders> struct FieldPack : Holders... {
  static constexpr std::size_t size = sizeof...(Holders);

  KOKKOS_FUNCTION FieldPack() = default;
  KOKKOS_FUNCTION FieldPack(const FieldPack &) = default;
  KOKKOS_FUNCTION FieldPack &operator=(const FieldPack &) = default;

  /// @brief Construct from existing holder instances (e.g., in tests)
  KOKKOS_FUNCTION FieldPack(const Holders &...holders) : Holders(holders)... {}

  /// @brief Construct all holders from a Kokkos scratch memory space.
  /// SFINAE guard prevents this overload from being selected when ScratchSpace
  /// is one of the Holders types (which would ambiguate with value
  /// constructor).
  template <typename ScratchSpace,
            std::enable_if_t<!std::disjunction_v<std::is_same<
                                 std::decay_t<ScratchSpace>, Holders>...>,
                             int> = 0>
  KOKKOS_FUNCTION FieldPack(const ScratchSpace &scratch)
      : Holders(scratch)... {}

  static std::size_t shmem_size() { return (Holders::shmem_size() + ...); }
};

} // namespace specfem::chunk_element
