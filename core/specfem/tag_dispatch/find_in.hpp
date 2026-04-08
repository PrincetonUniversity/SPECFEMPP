#pragma once

#include "is_valid.hpp"
#include "specfem/tags.hpp"
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace specfem::tag_dispatch {

namespace impl {

template <auto ValidCombinations, auto SearchCombination, std::size_t I>
struct valid_combination {
  static constexpr bool value = (ValidCombinations[I] == SearchCombination);
};

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

template <auto ValidCombinations, auto SearchCombination>
using find_in_temp =
    impl::find_in<ValidCombinations, SearchCombination,
                  std::make_index_sequence<ValidCombinations.size()> >;

template <auto ValidCombinations, auto SearchCombination>
using find_in_t =
    typename find_in_temp<ValidCombinations, SearchCombination>::type;

template <auto ValidCombinations, auto SearchCombination>
constexpr std::size_t find_in_v =
    find_in_t<ValidCombinations, SearchCombination>::value;

} // namespace specfem::tag_dispatch
