#pragma once

namespace specfem::assembly::fields_impl {

template <typename... AccessorTypes>
constexpr void check_accessor_compatibility() {
  constexpr auto MediumTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::medium_tag;
  static_assert(((AccessorTypes::medium_tag == MediumTag) && ...),
                "All accessors must have the same medium tag");
  constexpr auto DimensionTag =
      std::tuple_element_t<0, std::tuple<AccessorTypes...> >::dimension_tag;
  static_assert(((AccessorTypes::dimension_tag == DimensionTag) && ...),
                "All accessors must have the same dimension tag");
  static_assert((AccessorTypes::simd::using_simd && ...) ||
                    !(AccessorTypes::simd::using_simd || ...),
                "All accessors must be either SIMD or non-SIMD");
}

} // namespace specfem::assembly::fields_impl
