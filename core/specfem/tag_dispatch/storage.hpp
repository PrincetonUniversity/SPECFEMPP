#pragma once

#include "element_combinations.hpp"
#include "find_in.hpp"
#include "for_each.hpp"
#include "specfem/tags.hpp"
#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace specfem::tag_dispatch {

namespace impl {
// ── to_tuple helpers (Tags<...> → TagValueTuple) ─────────────────────────────

template <typename T> struct to_tuple;

template <auto... TagValues>
struct to_tuple<specfem::tags::Tags<TagValues...> > {
  constexpr static auto value =
      TagValueTuple<decltype(TagValues)...>{ TagValues... };
};

// ── combo_to_tags: TagValueTuple NTTP → Tags<...> instance ──────────────────
// Usage: combo_to_tags<element_combinations[I]>()

template <auto Combo, std::size_t... Is>
constexpr auto combo_to_tags_impl(std::index_sequence<Is...>) {
  return specfem::tags::Tags<Combo.template get<Is>()...>{};
}

template <auto Combo> constexpr auto combo_to_tags() {
  return combo_to_tags_impl<Combo>(
      std::make_index_sequence<std::decay_t<decltype(Combo)>::arity>{});
}

// ── Storage
// ───────────────────────────────────────────────────────────────────

} // namespace impl

template <typename T, typename ET> class Storage {

public:
  using type = T;
  static constexpr std::size_t size = ET::size;
  static constexpr auto element_combinations = ET::combos;

private:
  std::array<type, size> data_;

public:
  Storage() = default;

  template <typename Func> Storage(Func &&initializer) : data_() {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((([&] {
         using TagsType = std::decay_t<decltype(
             impl::combo_to_tags<element_combinations[Is]>())>;
         data_[Is] = initializer.template operator()<TagsType>();
       })()),
       ...);
    }(std::make_index_sequence<size>{});
  }

  template <
      typename TagsType,
      std::size_t Idx = find_in_v<ET::combos, impl::to_tuple<TagsType>::value> >
  constexpr const type &get() const {
    return data_[Idx];
  }

  template <typename... QueryTagTypes>
  requires((sizeof...(QueryTagTypes) > 0) &&
           ((std::is_enum_v<std::remove_cvref_t<QueryTagTypes> > ||
             std::is_same_v<std::remove_cvref_t<QueryTagTypes>, bool>) &&
            ...)) const type &get(QueryTagTypes &&...query_tags) const {
    const type *result = nullptr;
    specfem::tag_dispatch::for_each(ET{}, [&]<typename TagsType>() {
      if (!result && TagsType{}.has(query_tags...))
        result = &this->template get<TagsType>();
    });
    if (!result) {
      auto tag_to_str = [](const auto &t) -> std::string {
        using TagType = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<TagType, bool>) {
          return t ? "true" : "false";
        } else {
          using specfem::element::to_string;
          if constexpr (requires { to_string(t); }) {
            return to_string(t);
          } else {
            return std::to_string(
                static_cast<std::underlying_type_t<TagType>>(t));
          }
        }
      };
      std::string msg =
          "no matching element combination for queried tags: [";
      bool first = true;
      ((msg += (first ? (first = false, std::string{}) : std::string{ ", " }) +
               tag_to_str(query_tags)),
       ...);
      msg += "]";
      throw std::runtime_error(msg);
    }
    return *result;
  }
};

} // namespace specfem::tag_dispatch
