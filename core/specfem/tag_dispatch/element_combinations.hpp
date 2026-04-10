#pragma once

#include "is_valid.hpp"

#include <array>
#include <cstddef>
#include <tuple>

// ── Named tag-set types
// ─────────────────────────────────────────────────────── Each type is
// constrained to its own enum and exposes tag_enum + values. Usage (with using
// namespace tags):
//   specfem::tag_dispatch::element_combinations<dimensions<dim2>,
//                        media<elastic_psv, acoustic>,
//                        properties<isotropic>,
//                        attenuation<no_attenuation>,
//                        boundary<no_boundary, stacey>>

namespace specfem::tag_dispatch {

template <specfem::element::dimension_tag... Vs> struct dimension_set {
  using tag_enum = specfem::element::dimension_tag;
  static constexpr std::array<specfem::element::dimension_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

template <specfem::element::medium_tag... Vs> struct medium_set {
  using tag_enum = specfem::element::medium_tag;
  static constexpr std::array<specfem::element::medium_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

template <specfem::element::property_tag... Vs> struct property_set {
  using tag_enum = specfem::element::property_tag;
  static constexpr std::array<specfem::element::property_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

template <specfem::element::attenuation_tag... Vs> struct attenuation_set {
  using tag_enum = specfem::element::attenuation_tag;
  static constexpr std::array<specfem::element::attenuation_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

template <specfem::element::boundary_tag... Vs> struct boundary_set {
  using tag_enum = specfem::element::boundary_tag;
  static constexpr std::array<specfem::element::boundary_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

template <specfem::simulation::field_type... Vs> struct wavefield_set {
  using tag_enum = specfem::simulation::field_type;
  static constexpr std::array<specfem::simulation::field_type, sizeof...(Vs)>
      values{ { Vs... } };
};

// ── Helpers
// ───────────────────────────────────────────────────────────────────

namespace impl {

// Map TagValueTuple arity → the right validity predicate.
template <typename Tuple> constexpr bool is_valid(const Tuple &t) {
  if constexpr (Tuple::arity == 2)
    return is_valid_medium_combo(t);
  else if constexpr (Tuple::arity == 3)
    return is_valid_property_combo(t);
  else if constexpr (Tuple::arity == 4) {
    using T3 = decltype(t.template get<3>());
    if constexpr (std::is_same_v<T3, specfem::element::boundary_tag>)
      return is_valid_boundary_combo(t);
    else if constexpr (std::is_same_v<T3, specfem::element::attenuation_tag>)
      return is_valid_material_combo(t);
    else
      return false;
  } else if constexpr (Tuple::arity == 5)
    return is_valid_full_combo(t);
  else if constexpr (Tuple::arity == 6) {
    // Arity-6: (dim, medium, property, attenuation, boundary, wavefield)
    // Check the first 5 elements; wavefield (position 5) never invalidates
    return is_valid_full_combo(
        specfem::tag_dispatch::impl::TagValueTuple<
            decltype(t.template get<0>()), decltype(t.template get<1>()),
            decltype(t.template get<2>()), decltype(t.template get<3>()),
            decltype(t.template get<4>())>{
            t.template get<0>(), t.template get<1>(), t.template get<2>(),
            t.template get<3>(), t.template get<4>() });
  } else
    return false;
}

// ── Generic recursive count / fill ───────────────────────────────────────────
// Iterates over a std::tuple of arrays, recursing one array at a time,
// accumulating chosen values until the leaf where validity is checked.

template <typename Combo, std::size_t I = 0, typename ArrayTuple,
          typename... Chosen>
constexpr std::size_t count_combos(const ArrayTuple &arrs, Chosen... chosen) {
  if constexpr (I == std::tuple_size_v<ArrayTuple>)
    return is_valid(Combo{ chosen... }) ? 1 : 0;
  else {
    std::size_t n = 0;
    for (auto v : std::get<I>(arrs))
      n += count_combos<Combo, I + 1>(arrs, chosen..., v);
    return n;
  }
}

template <typename Combo, std::size_t N, std::size_t I = 0, typename ArrayTuple,
          typename... Chosen>
constexpr void fill_combos_helper(std::array<Combo, N> &out, std::size_t &idx,
                                  const ArrayTuple &arrs, Chosen... chosen) {
  if constexpr (I == std::tuple_size_v<ArrayTuple>) {
    if (is_valid(Combo{ chosen... }))
      out[idx++] = Combo{ chosen... };
  } else {
    for (auto v : std::get<I>(arrs))
      fill_combos_helper<Combo, N, I + 1>(out, idx, arrs, chosen..., v);
  }
}

template <typename Combo, std::size_t N, typename ArrayTuple>
constexpr std::array<Combo, N> fill_combos(const ArrayTuple &arrs) {
  std::array<Combo, N> out{};
  std::size_t idx = 0;
  fill_combos_helper<Combo, N, 0>(out, idx, arrs);
  return out;
}

} // namespace impl

// ── element_combinations
// ────────────────────────────────────────────────────── Single generic struct:
// accepts any number of named tag-set types (dimensions<>, media<>,
// properties<>, attenuation<>, boundary<>).

template <typename... NamedSets> struct element_combinations {
  using combo_type = impl::TagValueTuple<typename NamedSets::tag_enum...>;

private:
  static constexpr auto s_arrays = std::make_tuple(NamedSets::values...);

public:
  static constexpr std::size_t size =
      impl::count_combos<combo_type, 0>(s_arrays);
  static constexpr std::array<combo_type, size> combos =
      impl::fill_combos<combo_type, size>(s_arrays);
};

// ── operator* for building element_combinations from tag sets
// ───────────────── Enables: constexpr auto ET = DIMENSION_SET(dim2){} *
// MEDIUM_SET(...){} * ...;

// Base: two tag sets → element_combinations<A, B>
template <typename A, typename B>
constexpr auto operator*(A, B) -> element_combinations<A, B> {
  return {};
}

// Extension: element_combinations<...> * new tag set → larger
// element_combinations
template <typename... Existing, typename NewSet>
constexpr auto operator*(element_combinations<Existing...>, NewSet)
    -> element_combinations<Existing..., NewSet> {
  return {};
}

} // namespace specfem::tag_dispatch
