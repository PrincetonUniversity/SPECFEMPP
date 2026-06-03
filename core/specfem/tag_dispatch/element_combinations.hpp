#pragma once

#include "is_valid.hpp"
#include <array>
#include <cstddef>
#include <tuple>

namespace specfem::tag_dispatch {

/**
 * @brief Tag-set type for dimension values.
 *
 * Wraps a variadic list of `specfem::element::dimension_tag` values into a
 * named struct that can be composed with other tag-set types via `operator*`
 * to form an `element_combinations` type.
 *
 * @tparam Vs  One or more `dimension_tag` enumerators to include.
 *
 * @code
 * using DimSet = specfem::tag_dispatch::dimension_set<
 *     specfem::element::dimension_tag::dim2>;
 * @endcode
 */
template <specfem::element::dimension_tag... Vs> struct dimension_set {
  using tag_enum = specfem::element::dimension_tag;
  static constexpr std::array<specfem::element::dimension_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

/**
 * @brief Tag-set type for medium (wave-physics) values.
 *
 * @tparam Vs  One or more `medium_tag` enumerators to include.
 */
template <specfem::element::medium_tag... Vs> struct medium_set {
  using tag_enum = specfem::element::medium_tag;
  static constexpr std::array<specfem::element::medium_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

/**
 * @brief Tag-set type for material property values.
 *
 * @tparam Vs  One or more `property_tag` enumerators to include.
 */
template <specfem::element::property_tag... Vs> struct property_set {
  using tag_enum = specfem::element::property_tag;
  static constexpr std::array<specfem::element::property_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

/**
 * @brief Tag-set type for attenuation model values.
 *
 * @tparam Vs  One or more `attenuation_tag` enumerators to include.
 */
template <specfem::element::attenuation_tag... Vs> struct attenuation_set {
  using tag_enum = specfem::element::attenuation_tag;
  static constexpr std::array<specfem::element::attenuation_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

/**
 * @brief Tag-set type for boundary-condition values.
 *
 * @tparam Vs  One or more `boundary_tag` enumerators to include.
 */
template <specfem::element::boundary_tag... Vs> struct boundary_set {
  using tag_enum = specfem::element::boundary_tag;
  static constexpr std::array<specfem::element::boundary_tag, sizeof...(Vs)>
      values{ { Vs... } };
};

/**
 * @brief Tag-set type for MPI partition classification values.
 *
 * @tparam Vs  One or more `mpi_tag` enumerators to include.
 */
template <specfem::element::mpi_tag... Vs> struct mpi_set {
  using tag_enum = specfem::element::mpi_tag;
  static constexpr std::array<specfem::element::mpi_tag, sizeof...(Vs)> values{
    { Vs... }
  };
};

template <specfem::element_connections::type... Vs> struct connection_set {
  using tag_enum = specfem::element_connections::type;
  static constexpr std::array<specfem::element_connections::type, sizeof...(Vs)>
      values{ { Vs... } };
};

template <specfem::element_coupling::interface_tag... Vs> struct interface_set {
  using tag_enum = specfem::element_coupling::interface_tag;
  static constexpr std::array<specfem::element_coupling::interface_tag,
                              sizeof...(Vs)>
      values{ { Vs... } };
};

template <specfem::element_coupling::flux_scheme_tag... Vs>
struct flux_scheme_set {
  using tag_enum = specfem::element_coupling::flux_scheme_tag;
  static constexpr std::array<specfem::element_coupling::flux_scheme_tag,
                              sizeof...(Vs)>
      values{ { Vs... } };
};

/**
 * @brief Tag-set type for simulation wavefield field types.
 *
 * @tparam Vs  One or more `simulation::field_type` enumerators to include.
 */
template <specfem::simulation::field_type... Vs> struct wavefield_set {
  using tag_enum = specfem::simulation::field_type;
  static constexpr std::array<specfem::simulation::field_type, sizeof...(Vs)>
      values{ { Vs... } };
};

namespace impl {

/**
 * @brief Dispatch a `TagValueTuple` of any arity to the appropriate validity
 *        predicate.
 *
 * Dispatches by first-slot type:
 * - `field_type` first slot: wavefield-keyed combos, always valid.
 * - `dimension_tag` first slot: element combos, dispatched by arity 2–5.
 *   Trailing non-physical tags (wavefield, mpi) are recursively stripped.
 * - `element_connections::type` second slot: interface/coupling combos.
 *
 * @tparam Tuple  A `TagValueTuple` specialisation.
 * @param  t      The tuple to validate.
 * @return `true` if the combination is physically meaningful.
 */
template <typename Tuple> constexpr bool is_valid(const Tuple &t) {
  using T0 = decltype(t.template get<0>());

  // Wavefield-keyed combinations (e.g., wavefield_set * medium_set):
  // all combos are valid — no physics constraints on wavefield × medium.
  if constexpr (std::is_same_v<T0, specfem::simulation::field_type>) {
    return true;
  } else if constexpr (std::is_same_v<T0, specfem::element::dimension_tag>) {
    using T1 = decltype(t.template get<1>());

    if constexpr (std::is_same_v<T1, specfem::element::medium_tag>) {
      // Element combinations: dispatch by arity
      if constexpr (Tuple::arity == 2)
        return is_valid_medium_combo(t);
      else if constexpr (Tuple::arity == 3)
        return is_valid_property_combo(t);
      else if constexpr (Tuple::arity == 4) {
        using T3 = decltype(t.template get<3>());
        if constexpr (std::is_same_v<T3, specfem::element::boundary_tag>)
          return is_valid_boundary_combo(t);
        else if constexpr (std::is_same_v<T3,
                                          specfem::element::attenuation_tag>)
          return is_valid_material_combo(t);
        else
          return false;
      } else if constexpr (Tuple::arity >= 5) {
        // Recursively strip trailing non-physical tags (wavefield, mpi)
        using TLast = decltype(t.template get<Tuple::arity - 1>());
        if constexpr (non_physical_tag<TLast>)
          return is_valid(strip_last(t));
        else if constexpr (Tuple::arity == 5)
          return is_valid_full_combo(t);
        else
          return false;
      } else
        return false;

    } else if constexpr (std::is_same_v<T1,
                                        specfem::element_connections::type>) {
      // Interface / coupling combinations
      if constexpr (Tuple::arity == 3)
        return is_valid_interface_system(t);
      else if constexpr (Tuple::arity == 4)
        return is_valid_edge(t);
      else if constexpr (Tuple::arity == 5)
        return is_valid_edge_and_flux_scheme(t);
      else
        return false;
    } else
      return false;
  } else
    return false;
}

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

/**
 * @brief Fill `out` with valid combos by recursively iterating tag-value
 * arrays.
 *
 * Mirrors `count_combos` but writes each valid leaf combo into `out[idx++]`
 * instead of counting it.
 *
 * @tparam Combo       The `TagValueTuple` type stored in `out`.
 * @tparam N           Capacity of `out` (must equal the result of
 * `count_combos`).
 * @tparam I           Current recursion depth; defaults to 0.
 * @tparam ArrayTuple  `std::tuple` of `constexpr` tag-value arrays.
 * @tparam Chosen      Values chosen so far.
 * @param  out         Output array to fill.
 * @param  idx         Write cursor into `out`; incremented for each valid
 * combo.
 * @param  arrs        Tuple of arrays to iterate over.
 * @param  chosen      Tag values chosen in outer recursion levels.
 */
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

/**
 * @brief Build and return the full array of valid combos.
 *
 * Constructs a default-initialised `std::array<Combo, N>`, then calls
 * `fill_combos_helper` to populate it.
 *
 * @tparam Combo       The `TagValueTuple` type for each combo.
 * @tparam N           Number of valid combos (must match `count_combos`
 * result).
 * @tparam ArrayTuple  `std::tuple` of `constexpr` tag-value arrays.
 * @param  arrs        Tuple of arrays to iterate over.
 * @return Fully populated array of valid combos.
 */
template <typename Combo, std::size_t N, typename ArrayTuple>
constexpr std::array<Combo, N> fill_combos(const ArrayTuple &arrs) {
  std::array<Combo, N> out{};
  std::size_t idx = 0;
  fill_combos_helper<Combo, N, 0>(out, idx, arrs);
  return out;
}

} // namespace impl

/**
 * @brief Enumerates all **valid** element tag combinations from a set of named
 *        tag-set types.
 *
 * At compile time, `element_combinations` forms the Cartesian product of the
 * provided tag-set value arrays, filters it through the `is_valid_*_combo`
 * predicates, and stores only the valid tuples in `combos`.
 *
 * @tparam NamedSets  A pack of tag-set types (`dimension_set`, `medium_set`,
 *                    `property_set`, `attenuation_set`, `boundary_set`, or
 *                    `wavefield_set`), each contributing one slot to the combo.
 *
 * @par Members
 * - `combo_type`  The `TagValueTuple` type for a single combo (one slot per
 *                 `NamedSet`).
 * - `size`        Number of valid combos.
 * - `combos`      `constexpr std::array<combo_type, size>` of valid combos.
 *
 * @code
 * using ET = decltype(
 *     specfem::tag_dispatch::dimension_set<
 *         specfem::element::dimension_tag::dim2>{} *
 *     specfem::tag_dispatch::medium_set<
 *         specfem::element::medium_tag::elastic_psv,
 *         specfem::element::medium_tag::elastic_sh,
 *         specfem::element::medium_tag::acoustic>{} *
 *     specfem::tag_dispatch::property_set<
 *         specfem::element::property_tag::isotropic>{} *
 *     specfem::tag_dispatch::attenuation_set<
 *         specfem::element::attenuation_tag::none>{} *
 *     specfem::tag_dispatch::boundary_set<
 *         specfem::element::boundary_tag::none,
 *         specfem::element::boundary_tag::stacey>{});
 *
 * static_assert(ET::size > 0);
 * // ET::combos[0] = (dim2, elastic_psv, isotropic, none, none)
 * // ET::combos[1] = (dim2, elastic_psv, isotropic, none, stacey)
 * // ...
 * @endcode
 */
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

/**
 * @brief Compose two tag-set types into an `element_combinations<A, B>`.
 *
 * @code
 * auto et = specfem::tag_dispatch::dimension_set<dim2>{} *
 *           specfem::tag_dispatch::medium_set<elastic_psv>{};
 * @endcode
 */
template <typename A, typename B>
constexpr auto operator*(A, B) -> element_combinations<A, B> {
  return {};
}

/**
 * @brief Extend an existing `element_combinations` with one more tag-set type.
 *
 * Allows chaining multiple `operator*` calls to grow the combo type
 * incrementally:
 * @code
 * // Each * appends one more named set
 * auto et = dim_set{} * med_set{} * prop_set{} * att_set{} * bnd_set{};
 * @endcode
 */
template <typename... Existing, typename NewSet>
constexpr auto operator*(element_combinations<Existing...>, NewSet)
    -> element_combinations<Existing..., NewSet> {
  return {};
}

} // namespace specfem::tag_dispatch
