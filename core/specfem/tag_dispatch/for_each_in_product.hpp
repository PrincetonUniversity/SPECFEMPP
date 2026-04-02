#pragma once

#include "element_combinations.hpp"
#include "is_valid.hpp"
#include "specfem/tags.hpp"

#include <cstddef>

namespace specfem::tag_dispatch {

/**
 * for_each_in_product<ET>(lambda)
 * for_each_in_product<DimSet, MedSet, ...>(lambda)
 *
 * Calls lambda() with each valid Tags<v0, v1, ...> as a template argument.
 */

namespace impl {

// Unpack combo I from ET::combos into a single Tags<> instance and call f.
template <typename ET, std::size_t I, typename Func, std::size_t... Js>
void dispatch_one_impl(Func &f, std::index_sequence<Js...>) {
  constexpr auto c = ET::combos[I];
  using T = specfem::tags::Tags<c.template get<Js>()...>;
  f.template operator()<T>();
}

template <typename ET, std::size_t I, typename Func>
void dispatch_one(Func &f) {
  dispatch_one_impl<ET, I>(f,
                           std::make_index_sequence<ET::combo_type::arity>{});
}

template <typename ET, typename Func, std::size_t... Is>
void for_each_impl(Func &f, std::index_sequence<Is...>) {
  (dispatch_one<ET, Is>(f), ...);
}

} // namespace impl

// Explicit-type overload: for_each_in_product<ET>(func)
template <typename ET, typename Func> void for_each_in_product(Func &&f) {
  impl::for_each_impl<ET>(f, std::make_index_sequence<ET::size>{});
}

// Multi-tagset explicit-type overload: for_each_in_product<DimSet, MedSet,
// ...>(func) Requires >=2 tag-set types to avoid ambiguity with the single-ET
// overload above.
template <typename TagSet0, typename TagSet1, typename... TagSets,
          typename Func>
void for_each_in_product(Func &&f) {
  using ET =
      specfem::tag_dispatch::element_combinations<TagSet0, TagSet1, TagSets...>;
  impl::for_each_impl<ET>(f, std::make_index_sequence<ET::size>{});
}

} // namespace specfem::tag_dispatch
