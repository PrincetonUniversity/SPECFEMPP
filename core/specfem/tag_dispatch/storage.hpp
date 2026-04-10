#pragma once

#include "element_combinations.hpp"
#include "for_each.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace specfem::tag_dispatch {

namespace impl {

// ── to_tuple: Tags<TagValues...> → TagValueTuple NTTP (used by runtime get())

template <typename T> struct to_tuple;

template <auto... TagValues>
struct to_tuple<specfem::tags::Tags<TagValues...> > {
  constexpr static auto value =
      TagValueTuple<decltype(TagValues)...>{ TagValues... };
};

// ── combo_to_tags_t: maps (ET, ComboIdx, Arity) → Tags<...> type
// Uses typename ET + size_t index rather than `auto Combo` struct NTTP so that
// NVCC's EDG front-end can handle this without corrupting type resolution.

template <typename ET, std::size_t ComboIdx, std::size_t Arity,
          typename Seq = std::make_index_sequence<Arity> >
struct combo_to_tags;

template <typename ET, std::size_t ComboIdx, std::size_t Arity,
          std::size_t... Js>
struct combo_to_tags<ET, ComboIdx, Arity, std::index_sequence<Js...> > {
  using type = specfem::tags::Tags<ET::combos[ComboIdx].template get<Js>()...>;
};

template <typename ET, std::size_t ComboIdx, std::size_t Arity>
using combo_to_tags_t = typename combo_to_tags<ET, ComboIdx, Arity>::type;

// ── Policy types
// ──────────────────────────────────────────────────────────────
//
// TypePolicy<T>        — all slots hold the same type T (ignores TagsType)
// TemplatePolicy<Tmpl> — slot for TagsType holds Tmpl<TagsType>

template <typename T> struct TypePolicy {
  template <typename TagsType> using type = T;
};

template <template <typename> class Tmpl> struct TemplatePolicy {
  template <typename TagsType> using type = Tmpl<TagsType>;
};

// ── UnifiedSlot: one base class per valid combo
// ───────────────────────────────

template <typename TagsType, typename Policy> struct UnifiedSlot {
  typename Policy::template type<TagsType> value{};
};

// ── UnifiedStorage
// ────────────────────────────────────────────────────────────
//
// Inherits one UnifiedSlot per valid combo in ET.
// get<TagsType>() is a plain base-class upcast — zero overhead, GPU-safe.

template <typename Policy, typename ET,
          typename = std::make_index_sequence<ET::size> >
class UnifiedStorage;

template <typename Policy, typename ET, std::size_t... Is>
class UnifiedStorage<Policy, ET, std::index_sequence<Is...> >
    : private UnifiedSlot<combo_to_tags_t<ET, Is, ET::combo_type::arity>,
                          Policy>... {

public:
  static constexpr std::size_t size = ET::size;
  static constexpr auto element_combinations = ET::combos;

  UnifiedStorage() = default;

  template <typename Func>
  explicit UnifiedStorage(Func &&initializer)
      : UnifiedSlot<combo_to_tags_t<ET, Is, ET::combo_type::arity>, Policy>{
          initializer.template
          operator()<combo_to_tags_t<ET, Is, ET::combo_type::arity> >()
        }... {}

  // Compile-time get: plain base-class upcast, zero overhead, GPU-safe
  template <typename TagsType> KOKKOS_INLINE_FUNCTION auto &get() {
    return UnifiedSlot<TagsType, Policy>::value;
  }

  template <typename TagsType> KOKKOS_INLINE_FUNCTION const auto &get() const {
    return UnifiedSlot<TagsType, Policy>::value;
  }

  // Runtime get: linear scan over all combos, host-only.
  // Only available for TypePolicy (homogeneous) stores where all slots share
  // type T — needed by get_sources_on_host/device.
  template <typename... QueryTagTypes>
  requires(
      std::is_same_v<Policy,
                     TypePolicy<typename Policy::template type<
                         combo_to_tags_t<ET, 0, ET::combo_type::arity> > > > &&
      (sizeof...(QueryTagTypes) > 0) &&
      ((std::is_enum_v<std::remove_cvref_t<QueryTagTypes> > ||
        std::is_same_v<std::remove_cvref_t<QueryTagTypes>, bool>) &&
       ...)) const auto &get(QueryTagTypes &&...query_tags) const {
    using SlotType = typename Policy::template type<
        combo_to_tags_t<ET, 0, ET::combo_type::arity> >;
    const SlotType *result = nullptr;
    specfem::tag_dispatch::for_each(ET{}, [&]<typename TagsType>() {
      if (!result && TagsType{}.has(query_tags...))
        result = &this->template get<TagsType>();
    });
    if (!result) {
      std::string tags_str;
      ((tags_str += (tags_str.empty() ? "" : ", ") +
                    specfem::element::to_string(query_tags)),
       ...);
      throw std::runtime_error(
          "no matching element combination for queried tags: [" + tags_str +
          "]");
    }
    return *result;
  }

  template <typename P1, typename P2, typename ET2, std::size_t... Js>
  friend void
  deep_copy(UnifiedStorage<P1, ET2, std::index_sequence<Js...> > &dest,
            const UnifiedStorage<P2, ET2, std::index_sequence<Js...> > &src);
};

// deep_copy: supports dest and src with different policies
// (e.g. device view type vs host mirror view)
template <typename Policy1, typename Policy2, typename ET, std::size_t... Is>
void deep_copy(
    UnifiedStorage<Policy1, ET, std::index_sequence<Is...> > &dest,
    const UnifiedStorage<Policy2, ET, std::index_sequence<Is...> > &src) {
  (Kokkos::deep_copy(
       static_cast<UnifiedSlot<combo_to_tags_t<ET, Is, ET::combo_type::arity>,
                               Policy1> &>(dest)
           .value,
       static_cast<const UnifiedSlot<
           combo_to_tags_t<ET, Is, ET::combo_type::arity>, Policy2> &>(src)
           .value),
   ...);
}

} // namespace impl

// ── Public aliases
// ────────────────────────────────────────────────────────────
//
// Storage<T, ET>          — homogeneous: every slot holds a T
// TypedStorage<Tmpl, ET>  — heterogeneous: slot for TagsType holds
// Tmpl<TagsType>

template <typename T, typename ET>
using Storage = impl::UnifiedStorage<impl::TypePolicy<T>, ET>;

template <template <typename> class Tmpl, typename ET>
using TypedStorage = impl::UnifiedStorage<impl::TemplatePolicy<Tmpl>, ET>;

} // namespace specfem::tag_dispatch
