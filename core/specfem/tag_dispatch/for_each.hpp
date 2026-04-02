#pragma once

#include "element_combinations.hpp"
#include "is_valid.hpp"
#include "specfem/tags.hpp"

#include <cstddef>

/**
 * specfem::tag_dispatch::for_each(et_value, lambda)
 *
 * Calls lambda() with each valid Tags<v0, v1, ...> as a template argument.
 * et_value is any element_combinations<...> instance, e.g.:
 *   specfem::tag_dispatch::for_each(DIMENSION_T(dim2) * MEDIUM_T(elastic_psv),
 * func);
 */

namespace specfem::tag_dispatch {

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

template <typename... Sets, typename Func>
void for_each(element_combinations<Sets...>, Func &&f) {
  using ET = element_combinations<Sets...>;
  impl::for_each_impl<ET>(f, std::make_index_sequence<ET::size>{});
}

} // namespace specfem::tag_dispatch
