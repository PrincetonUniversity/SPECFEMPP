#pragma once

#include "element_combinations.hpp"
#include "find_in.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <type_traits>

namespace specfem::tag_dispatch {

namespace impl {
// ── to_tuple helpers (Tags<...> → TagValueTuple) ─────────────────────────────

template <typename T> struct to_tuple;

template <auto... TagValues>
struct to_tuple<specfem::tags::Tags<TagValues...> > {
  constexpr static auto value =
      TagValueTuple<decltype(TagValues)...>{ TagValues... };
};

// ── Type trait: has_host_mirror
// ─────────────────────────────────────────────── Detects if T has a nested
// typedef HostMirror (e.g., Kokkos::View)

template <typename T, typename = void>
struct has_host_mirror : std::false_type {};

template <typename T>
struct has_host_mirror<T, std::void_t<typename T::HostMirror> >
    : std::true_type {};

// Forward declaration of Storage
template <typename T, typename ET> class Storage;

// ── Base class that conditionally provides HostMirror typedef
// ─────────────────

template <typename T, typename ET, bool = has_host_mirror<T>::value>
struct StorageHostMirrorBase {};

template <typename T, typename ET> struct StorageHostMirrorBase<T, ET, true> {
  using HostMirror = Storage<typename T::HostMirror, ET>;
};

// ── Storage
// ───────────────────────────────────────────────────────────────────

template <typename T, typename ET>
class Storage : public StorageHostMirrorBase<T, ET> {

public:
  using type = T;
  static constexpr std::size_t size = ET::size;
  static constexpr auto element_combinations = ET::combos;

private:
  using UnderlyingArray = std::array<type, size>;
  UnderlyingArray data_;

  const type &operator()(std::size_t i) const { return data_[i]; }

public:
  Storage() = default;

  template <typename Func> Storage(Func &&initializer) : data_() {
    for (std::size_t i = 0; i < size; ++i) {
      data_[i] = type(initializer(element_combinations[i]));
    }
  }

  template <typename TagsType,
            std::size_t Idx = find_in_v<ET::combos, to_tuple<TagsType>::value> >
  constexpr const type &get() const {
    return data_[Idx];
  }

  template <typename C, typename D, typename S>
  friend void deep_copy(Storage<D, C> &dest, const Storage<S, C> &src);
};

template <typename C, typename D, typename S>
void deep_copy(Storage<D, C> &dest, const Storage<S, C> &src) {
  for (std::size_t i = 0; i < C::size; ++i) {
    Kokkos::deep_copy(dest(i), src(i));
  }
}

// ── CTAD guide: deduce Storage<T,
// specfem::tag_dispatch::element_combinations<NamedSets...>> ───────── Usage:
// Storage s(value, dimensions<dim2>{}, media<elastic_psv>{},
// ...);

template <typename T, typename... NamedSets>
Storage(T, NamedSets...)
    -> Storage<T, specfem::tag_dispatch::element_combinations<NamedSets...> >;

} // namespace impl

template <typename T, typename ET> using Storage = impl::Storage<T, ET>;

} // namespace specfem::tag_dispatch
