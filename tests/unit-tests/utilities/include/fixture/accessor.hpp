#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/element_coupling/tags.hpp"
#include <type_traits>

namespace specfem::test_fixture {

namespace endow_accessor_impl {

/** @brief Used to capture a "null" type, since SFINAE doesn't work for this. */
struct Unset {};

#define declare_check(value_to_use)                                            \
                                                                               \
  template <typename T1, typename T2, typename = void> struct value_to_use {   \
    static constexpr bool has_value = false;                                   \
    static constexpr Unset value = {};                                         \
  };                                                                           \
  namespace has {                                                              \
  template <typename T2, typename = void>                                      \
  struct value_to_use : std::false_type {};                                    \
  template <typename T2>                                                       \
  struct value_to_use<T2,                                                      \
                      std::enable_if_t<T2::value_to_use == T2::value_to_use>>  \
      : std::true_type {};                                                     \
  }                                                                            \
  template <typename T1, typename T2>                                          \
  struct value_to_use<T1, T2,                                                  \
                      std::enable_if_t<has::value_to_use<T2>::value>> {        \
    static constexpr auto value = T2::value_to_use;                            \
    static constexpr bool has_value = true;                                    \
  };                                                                           \
  template <typename T1, typename T2>                                          \
  struct value_to_use<T1, T2,                                                  \
                      std::enable_if_t<has::value_to_use<T1>::value &&         \
                                       !has::value_to_use<T2>::value>> {       \
    static constexpr auto value = T1::value_to_use;                            \
    static constexpr bool has_value = true;                                    \
  };

declare_check(dimension_tag);
declare_check(interface_tag);
declare_check(accessor_type);
declare_check(boundary_tag);
declare_check(flux_scheme_tag);
declare_check(using_simd);

#undef declare_check

} // namespace endow_accessor_impl

template <typename OriginalView, typename AccessorType>
struct EndowAccessor : public OriginalView, public AccessorType {
  using OriginalView::OriginalView;

  // disambiguation
  static constexpr auto dimension_tag =
      endow_accessor_impl::dimension_tag<OriginalView, AccessorType>::value;
  static constexpr auto interface_tag =
      endow_accessor_impl::interface_tag<OriginalView, AccessorType>::value;
  static constexpr auto accessor_type =
      endow_accessor_impl::accessor_type<OriginalView, AccessorType>::value;
  static constexpr auto boundary_tag =
      endow_accessor_impl::boundary_tag<OriginalView, AccessorType>::value;
  static constexpr auto flux_scheme_tag =
      endow_accessor_impl::flux_scheme_tag<OriginalView, AccessorType>::value;
  static constexpr auto using_simd =
      endow_accessor_impl::using_simd<OriginalView, AccessorType>::value;
};
} // namespace specfem::test_fixture
