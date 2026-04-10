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

/**
 * @brief Convert a `Tags<TagValues...>` type to a `TagValueTuple` NTTP value.
 *
 * Used internally by the runtime `get()` overload to compare a runtime tuple
 * against compile-time combo entries.
 *
 * @tparam T  A `specfem::tags::Tags<...>` specialisation.
 */
template <typename T> struct to_tuple;

template <auto... TagValues>
struct to_tuple<specfem::tags::Tags<TagValues...> > {
  constexpr static auto value =
      TagValueTuple<decltype(TagValues)...>{ TagValues... };
};

/**
 * @brief Map an `element_combinations` type and a combo index to the
 *        corresponding `specfem::tags::Tags<...>` type.
 *
 * Unpacks the `Arity` tag values from `ET::combos[ComboIdx]` into a
 * `Tags<v0, v1, ..., v_{Arity-1}>` type.
 *
 * @tparam ET        `element_combinations` type.
 * @tparam ComboIdx  Zero-based index into `ET::combos`.
 * @tparam Arity     Number of slots per combo (`ET::combo_type::arity`).
 * @tparam Seq       Index sequence `0, 1, ..., Arity-1` (defaulted).
 */
template <typename ET, std::size_t ComboIdx, std::size_t Arity,
          typename Seq = std::make_index_sequence<Arity> >
struct combo_to_tags;

template <typename ET, std::size_t ComboIdx, std::size_t Arity,
          std::size_t... Js>
struct combo_to_tags<ET, ComboIdx, Arity, std::index_sequence<Js...> > {
  using type = specfem::tags::Tags<ET::combos[ComboIdx].template get<Js>()...>;
};

/** @brief Alias for the `Tags<...>` type at combo index `ComboIdx` in `ET`. */
template <typename ET, std::size_t ComboIdx, std::size_t Arity>
using combo_to_tags_t = typename combo_to_tags<ET, ComboIdx, Arity>::type;

/**
 * @brief Storage policy where every combo slot holds the same type `T`.
 *
 * `TypePolicy<T>::type<TagsType>` is always `T`, regardless of `TagsType`.
 * Used by `Storage<T,ET>` for homogeneous containers.
 *
 * @tparam T  The uniform value type for all slots.
 */
template <typename T> struct TypePolicy {
  template <typename TagsType> using type = T;
};

/**
 * @brief Storage policy where each combo slot holds `Tmpl<TagsType>`.
 *
 * `TemplatePolicy<Tmpl>::type<TagsType>` is `Tmpl<TagsType>`, yielding a
 * different concrete type per combo.  Used by `TypedStorage<Tmpl,ET>`.
 *
 * @tparam Tmpl  A class template parameterised by a `Tags<...>` type.
 */
template <template <typename> class Tmpl> struct TemplatePolicy {
  template <typename TagsType> using type = Tmpl<TagsType>;
};

/**
 * @brief A single inheritance slot for one combo in a `UnifiedStorage`.
 *
 * Holds `value` of type `Policy::type<TagsType>`.  Accessed via a
 * `static_cast` inside `UnifiedStorage::get<TagsType>()`.
 *
 * @tparam TagsType  The `specfem::tags::Tags<...>` type identifying this slot.
 * @tparam Policy    `TypePolicy<T>` or `TemplatePolicy<Tmpl>`.
 */
template <typename TagsType, typename Policy> struct UnifiedSlot {
  typename Policy::template type<TagsType> value{};
};

/**
 * @brief Per-combo storage container implemented via multiple inheritance.
 *
 * `UnifiedStorage` privately inherits one `UnifiedSlot<TagsType, Policy>` per
 * valid combo in `ET`.  `get<TagsType>()` is a plain base-class upcast — zero
 * overhead and device-safe.
 *
 * Construction accepts an optional callable `initializer` that is invoked once
 * per slot as `initializer.template operator()<TagsType>()` and must return the
 * slot's value type.
 *
 * @tparam Policy  Storage policy (`TypePolicy<T>` or `TemplatePolicy<Tmpl>`).
 * @tparam ET      `element_combinations` type providing `combos` and `size`.
 */
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

  /** @brief Default-construct all slots using the default constructor of their
   * value type. */
  UnifiedStorage() = default;

  /**
   * @brief Construct each slot by calling `initializer.operator()<TagsType>()`.
   *
   * The initializer is invoked once per valid combo with the combo's
   * `Tags<...>` type as a template argument.  It must return a value
   * convertible to `Policy::type<TagsType>`.
   *
   * @tparam Func  Generic callable with `template <typename T> auto
   * operator()()`.
   * @param  initializer  The initializer functor.
   *
   * @code
   * Storage<Kokkos::View<double*>, ET> s(
   *     []<typename TagsType>() {
   *         return Kokkos::View<double*>("field", n);
   *     });
   * @endcode
   */
  template <typename Func>
  explicit UnifiedStorage(Func &&initializer)
      : UnifiedSlot<combo_to_tags_t<ET, Is, ET::combo_type::arity>, Policy>{
          initializer.template
          operator()<combo_to_tags_t<ET, Is, ET::combo_type::arity> >()
        }... {}

  /**
   * @brief Compile-time slot access (mutable).
   *
   * Performs a plain base-class upcast; zero runtime overhead.  A `TagsType`
   * that is not a valid combo in `ET` produces a hard compile error.
   *
   * @tparam TagsType  `specfem::tags::Tags<...>` identifying the desired slot.
   * @return  Reference to the stored value.
   */
  template <typename TagsType> KOKKOS_INLINE_FUNCTION auto &get() {
    return UnifiedSlot<TagsType, Policy>::value;
  }

  /**
   * @brief Compile-time slot access (const).
   *
   * @tparam TagsType  `specfem::tags::Tags<...>` identifying the desired slot.
   * @return  Const reference to the stored value.
   */
  template <typename TagsType> KOKKOS_INLINE_FUNCTION const auto &get() const {
    return UnifiedSlot<TagsType, Policy>::value;
  }

  /**
   * @brief Runtime slot access by querying individual enum tag values (host
   * only, homogeneous stores only).
   *
   * Iterates over all combos using `for_each` and returns the first slot whose
   * `Tags` type satisfies `Tags::has(query_tags...)`.  Only available when
   * `Policy` is `TypePolicy<T>` (all slots share the same type).
   *
   * @tparam QueryTagTypes  Enum types (or `bool`) of the query values.
   * @param  query_tags     Tag values to match against.
   * @return  Const reference to the matching slot's value.
   * @throws std::runtime_error if no combo matches the provided tags.
   */
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

/**
 * @brief Perform a `Kokkos::deep_copy` for every slot across two
 *        `UnifiedStorage` instances with potentially different policies.
 *
 * Useful for copying between host-mirror and device storage when the slot type
 * differs between policies (e.g. `Kokkos::View` vs its host mirror).
 *
 * @tparam Policy1  Destination storage policy.
 * @tparam Policy2  Source storage policy.
 * @tparam ET       Shared `element_combinations` type.
 * @param  dest     Destination storage.
 * @param  src      Source storage.
 */
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

/**
 * @brief Homogeneous per-combo storage: every combo slot holds a `T`.
 *
 * Alias for `impl::UnifiedStorage<impl::TypePolicy<T>, ET>`.
 *
 * @tparam T   The uniform value type stored for every valid combo.
 * @tparam ET  `element_combinations` type.
 *
 * @code
 * specfem::tag_dispatch::Storage<Kokkos::View<double*>, ET> s(
 *     []<typename TagsType>() {
 *         return Kokkos::View<double*>("field", n_elems);
 *     });
 * auto& view = s.get<T>();
 * @endcode
 */
template <typename T, typename ET>
using Storage = impl::UnifiedStorage<impl::TypePolicy<T>, ET>;

/**
 * @brief Heterogeneous per-combo storage: combo `TagsType` holds
 * `Tmpl<TagsType>`.
 *
 * Alias for `impl::UnifiedStorage<impl::TemplatePolicy<Tmpl>, ET>`.
 *
 * @tparam Tmpl  Class template parameterised by a `Tags<...>` type.
 * @tparam ET    `element_combinations` type.
 *
 * @code
 * template <typename TagsType> struct Kernel { void run(); };
 * specfem::tag_dispatch::TypedStorage<Kernel, ET> kernels(
 *     []<typename TagsType>() { return Kernel<TagsType>{}; });
 * auto& k = kernels.get<T>();  // Kernel<T>&
 * @endcode
 */
template <template <typename> class Tmpl, typename ET>
using TypedStorage = impl::UnifiedStorage<impl::TemplatePolicy<Tmpl>, ET>;

} // namespace specfem::tag_dispatch
