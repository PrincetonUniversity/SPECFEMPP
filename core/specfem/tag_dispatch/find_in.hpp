#pragma once

#include "is_valid.hpp"
#include "specfem/tags.hpp"
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace specfem::tag_dispatch {

namespace impl {

/**
 * @brief Test whether `ValidCombinations[I] == SearchCombination`.
 *
 * @tparam ValidCombinations  Constexpr array of combos (e.g. `ET::combos`).
 * @tparam SearchCombination  The combo value to search for.
 * @tparam I                  The index to test.
 */
template <auto ValidCombinations, auto SearchCombination, std::size_t I>
struct valid_combination {
  static constexpr bool value = (ValidCombinations[I] == SearchCombination);
};

/**
 * @brief Recursive linear search over `ValidCombinations`.
 *
 * Primary template (open index sequence) delegates to `valid_combination<I>`;
 * if true it resolves to `std::integral_constant<std::size_t, I>`, otherwise
 * it recurses with the remaining indices.
 *
 * Specialisation with an empty index sequence resolves `::type` to `void`
 * (combination not found).
 *
 * @tparam ValidCombinations  Constexpr array of combos.
 * @tparam SearchCombination  Combo value to find.
 * @tparam IndexSeq           Remaining indices to search.
 */
template <auto ValidCombinations, auto SearchCombination, typename IndexSeq>
struct find_in;

template <auto ValidCombinations, auto SearchCombination, std::size_t I,
          std::size_t... Is>
struct find_in<ValidCombinations, SearchCombination,
               std::index_sequence<I, Is...> > {
  using type = std::conditional_t<
      (valid_combination<ValidCombinations, SearchCombination, I>::value),
      std::integral_constant<std::size_t, I>,
      typename find_in<ValidCombinations, SearchCombination,
                       std::index_sequence<Is...> >::type>;
};

template <auto ValidCombinations, auto SearchCombination>
struct find_in<ValidCombinations, SearchCombination, std::index_sequence<> > {
  using type = void; // Not found case
};

} // namespace impl

/**
 * @brief Convenience alias: resolves to the `find_in` instance over the full
 *        `ValidCombinations` array.
 *
 * @tparam ValidCombinations  Constexpr combo array (e.g. `ET::combos`).
 * @tparam SearchCombination  Combo value to find.
 */
template <auto ValidCombinations, auto SearchCombination>
using find_in_temp =
    impl::find_in<ValidCombinations, SearchCombination,
                  std::make_index_sequence<ValidCombinations.size()> >;

/**
 * @brief Type alias that resolves to
 *        `std::integral_constant<std::size_t, I>` for the found index, or
 *        `void` if not found.
 *
 * @tparam ValidCombinations  Constexpr combo array.
 * @tparam SearchCombination  Combo value to find.
 */
template <auto ValidCombinations, auto SearchCombination>
using find_in_t =
    typename find_in_temp<ValidCombinations, SearchCombination>::type;

/**
 * @brief Constexpr index of `SearchCombination` inside `ValidCombinations`.
 *
 * Produces a compile error (via `::value` on `void`) when the combination is
 * not present.
 *
 * @tparam ValidCombinations  Constexpr combo array.
 * @tparam SearchCombination  Combo value to find.
 */
template <auto ValidCombinations, auto SearchCombination>
constexpr std::size_t find_in_v =
    find_in_t<ValidCombinations, SearchCombination>::value;

} // namespace specfem::tag_dispatch
