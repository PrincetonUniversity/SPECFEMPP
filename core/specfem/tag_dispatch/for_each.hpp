#pragma once

#include "element_combinations.hpp"
#include "is_valid.hpp"
#include "specfem/tags.hpp"

#include <cstddef>

namespace specfem::tag_dispatch {

namespace impl {

// Unpack combo I from ET::combos into a single Tags<> instance and call f.
/**
 * @brief Unpack combo `I` from `ET::combos` into a `Tags<...>` type and invoke
 *        `f.operator()<Tags<...>>()`.
 *
 * @tparam ET    `element_combinations` type.
 * @tparam I     Index into `ET::combos`.
 * @tparam Func  Callable with a `template operator()<T>()` overload.
 * @tparam Js    Index pack `0, 1, ..., arity-1` for unpacking the combo slots.
 */
template <typename ET, std::size_t I, typename Func, std::size_t... Js>
void dispatch_one_impl(Func &f, std::index_sequence<Js...>) {
  constexpr auto c = ET::combos[I];
  using T = specfem::tags::Tags<c.template get<Js>()...>;
  f.template operator()<T>();
}

/**
 * @brief Dispatch combo `I`: build the index sequence for the combo's arity and
 *        delegate to `dispatch_one_impl`.
 */
template <typename ET, std::size_t I, typename Func>
void dispatch_one(Func &f) {
  dispatch_one_impl<ET, I>(f,
                           std::make_index_sequence<ET::combo_type::arity>{});
}

/**
 * @brief Expand `ET::size` indices and call `dispatch_one` for each.
 */
template <typename ET, typename Func, std::size_t... Is>
void for_each_impl(Func &f, std::index_sequence<Is...>) {
  (dispatch_one<ET, Is>(f), ...);
}

} // namespace impl

/**
 * @brief Iterate over all valid element combinations in `ET`, calling
 *        `f.operator()<TagsType>()` for each.
 *
 * @tparam Sets  Tag-set types of the `element_combinations`.
 * @tparam Func  Generic callable; must provide
 *               `template <typename TagsType> void operator()()`.
 *
 * @param f  The callable to invoke per combo.
 *
 * @code
 * specfem::tag_dispatch::for_each(ET{}, []<typename TagsType>() {
 *     process<TagsType>();
 * });
 * @endcode
 */
template <typename... Sets, typename Func>
void for_each(element_combinations<Sets...>, Func &&f) {
  using ET = element_combinations<Sets...>;
  impl::for_each_impl<ET>(f, std::make_index_sequence<ET::size>{});
}

} // namespace specfem::tag_dispatch
