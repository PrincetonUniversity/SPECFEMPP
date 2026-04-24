#pragma once

#include "specfem/chunk_element/displacement.hpp"
#include "specfem/chunk_element/velocity.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace specfem::chunk_element {

/**
 * @brief Variadic field pack using multiple inheritance.
 *
 * Each `Fields` type becomes a public base class, so `FieldPack<F, G>` IS-A F
 * AND IS-A G. Access individual fields via `static_cast<F &>(pack)` or
 * `static_cast<const F &>(pack)`.
 *
 * All fields must share the same dimension_tag, medium_tag, ngll, nelements,
 * and using_simd.
 *
 * @tparam Fields... Chunk element field types (e.g., displacement<...>,
 *                   velocity<...>)
 */
template <typename... Fields> struct FieldPack : public Fields... {
  static constexpr std::size_t size = sizeof...(Fields);

  using first_field = std::tuple_element_t<0, std::tuple<Fields...> >;

  static_assert(
      (std::is_same_v<std::integral_constant<specfem::element::dimension_tag,
                                             Fields::dimension_tag>,
                      std::integral_constant<specfem::element::dimension_tag,
                                             first_field::dimension_tag> > &&
       ...),
      "All Fields in FieldPack must have the same dimension_tag");

  static_assert(
      (std::is_same_v<std::integral_constant<specfem::element::medium_tag,
                                             Fields::medium_tag>,
                      std::integral_constant<specfem::element::medium_tag,
                                             first_field::medium_tag> > &&
       ...),
      "All Fields in FieldPack must have the same medium_tag");

  static_assert(
      (std::is_same_v<std::integral_constant<int, Fields::ngll>,
                      std::integral_constant<int, first_field::ngll> > &&
       ...),
      "All Fields in FieldPack must have the same ngll");

  static_assert(
      (std::is_same_v<std::integral_constant<int, Fields::nelements>,
                      std::integral_constant<int, first_field::nelements> > &&
       ...),
      "All Fields in FieldPack must have the same nelements");

  static_assert(
      (std::is_same_v<std::integral_constant<bool, Fields::using_simd>,
                      std::integral_constant<bool, first_field::using_simd> > &&
       ...),
      "All Fields in FieldPack must have the same using_simd");

  KOKKOS_FUNCTION FieldPack() = default;
  KOKKOS_FUNCTION FieldPack(const FieldPack &) = default;
  KOKKOS_FUNCTION FieldPack &operator=(const FieldPack &) = default;

  /// @brief Construct from individual field instances.
  KOKKOS_FUNCTION explicit FieldPack(const Fields &...fields)
      : Fields(fields)... {}

  /// @brief Construct from Kokkos scratch memory space.
  ///
  /// Each base-class constructor is called in declaration order; Kokkos view
  /// constructors advance the internal scratch pointer sequentially, so fields
  /// do not overlap. The SFINAE guard ensures this overload is not selected
  /// when S is one of the Fields types.
  template <typename S,
            std::enable_if_t<(!std::is_same_v<std::decay_t<S>, Fields> && ...),
                             int> = 0>
  KOKKOS_FUNCTION FieldPack(const S &scratch) : Fields(scratch)... {}

  /// @brief Direct index access is deleted — cast to the desired base type.
  template <typename... Indices>
  KOKKOS_FUNCTION auto operator()(Indices...) const = delete;

  /// @brief Shared memory size required for all fields combined.
  static std::size_t shmem_size() { return (Fields::shmem_size() + ... + 0); }
};

/**
 * @brief Selects the appropriate FieldPack for the stiffness kernel.
 *
 * - `attenuation_tag::none`               → FieldPack<displacement<...>>
 * - `attenuation_tag::constant_isotropic` → FieldPack<displacement<...>,
 *                                                      velocity<...>>
 */
template <specfem::element::attenuation_tag AttenuationTag, int ChunkSize,
          int NGLL, specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
using stiffness_field_pack = std::conditional_t<
    AttenuationTag == specfem::element::attenuation_tag::none,
    FieldPack<displacement<ChunkSize, NGLL, DimensionTag, MediumTag, UseSIMD> >,
    FieldPack<displacement<ChunkSize, NGLL, DimensionTag, MediumTag, UseSIMD>,
              velocity<ChunkSize, NGLL, DimensionTag, MediumTag, UseSIMD> > >;

} // namespace specfem::chunk_element
