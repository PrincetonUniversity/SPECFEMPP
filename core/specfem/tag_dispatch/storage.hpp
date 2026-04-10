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
    std::size_t idx = size;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      (
          [&] {
            if (idx == size) {
              using TagsType = std::decay_t<decltype(
                  impl::combo_to_tags<element_combinations[Is]>())>;
              if (TagsType{}.has(query_tags...))
                idx = Is;
            }
          }(),
          ...);
    }(std::make_index_sequence<size>{});
    if (idx == size) {
      std::string tags_str;
      auto tag_to_str = [](auto tag) -> std::string {
        using specfem::element::to_string;
        using specfem::element_connections::to_string;
        using specfem::element_coupling::to_string;
        return to_string(tag);
      };
      ((tags_str += (tags_str.empty() ? "" : ", ") + tag_to_str(query_tags)),
       ...);
      throw std::runtime_error(
          "no matching element combination for queried tags: [" + tags_str +
          "]");
    }
    return data_[idx];
  }
};

/**
 * @brief Deep-copy customization point for use by @ref mirror_and_copy.
 *
 * The default implementation forwards to ``Kokkos::deep_copy``.  Custom view
 * types (e.g.\ multi-view structs) should provide an overload in their own
 * header *before* the point of instantiation to override this behaviour.
 */
template <typename Dst, typename Src> void deep_copy(Dst &&dst, Src &&src) {
  Kokkos::deep_copy(std::forward<Dst>(dst), std::forward<Src>(src));
}

/**
 * @brief Mirror-view creation customization point for use by
 *        @ref mirror_and_copy.
 *
 * The default implementation handles plain ``Kokkos::View`` types: it
 * allocates a new view in @p Space with the same data type, layout, and
 * extents as @p src.  Custom view types should provide an overload in their
 * own header *before* the point of instantiation.
 */
template <typename Space, typename SrcView>
auto create_mirror(Space, const SrcView &src) {
  return Kokkos::create_mirror_view(typename Space::memory_space{}, src);
}

/**
 * @brief Mirror a @ref Storage into a destination memory space, analogous to
 *        ``Kokkos::create_mirror_view``.
 *
 * If the memory space of @p Space is identical to the memory space of
 * @p SrcView (i.e.\ `std::is_same_v<typename Space::memory_space,
 * typename SrcView::memory_space>` is @c true), @p src_storage is returned
 * unchanged — no allocation or copy is performed.
 *
 * Otherwise, for each tag combination in the element-type set @p ET, a new
 * view is created via @ref create_mirror and the data is copied via
 * @ref deep_copy.  Both functions are customization points: custom view types
 * may provide overloads in their own headers.  The typical usage is to mirror
 * a host-space storage onto the device after it has been populated on the host:
 *
 * @code{.cpp}
 * HostStorage h_store{ initializer };
 * auto d_store = specfem::tag_dispatch::mirror_and_copy(
 *     Kokkos::DefaultExecutionSpace{}, h_store);
 * @endcode
 *
 * @tparam Space      Destination execution-space type.  Must expose a
 *                    ``memory_space`` typedef (satisfied by any Kokkos
 *                    execution space or device type).
 * @tparam SrcView    View type of the source storage.  Must expose a
 *                    ``memory_space`` typedef.  Deduced from @p src_storage.
 * @tparam ET         Element-type set that parameterises the storage.
 *                    Deduced from @p src_storage.
 *
 * @param src_storage Source storage to mirror.
 * @return            If the memory spaces match, returns @p src_storage as-is.
 *                    Otherwise returns a new @ref Storage whose view type is
 *                    the return type of
 *                    ``create_mirror<Space>(src_entry)``, populated by
 *                    calling ``deep_copy(dst_entry, src_entry)`` for each
 *                    tag combination.
 */
template <typename Space, typename SrcView, typename ET>
auto mirror_and_copy(Space, const Storage<SrcView, ET> &src_storage) {
  if constexpr (std::is_same_v<typename Space::memory_space,
                               typename SrcView::memory_space>) {
    return src_storage;
  } else {
    using DstView = decltype(
        create_mirror(std::declval<Space>(), std::declval<const SrcView &>()));
    return Storage<DstView, ET>{ [&]<typename TagsType>() -> DstView {
      const auto &src = src_storage.template get<TagsType>();
      auto dst = create_mirror(Space{}, src);
      deep_copy(dst, src);
      return dst;
    } };
  }
}

} // namespace specfem::tag_dispatch
