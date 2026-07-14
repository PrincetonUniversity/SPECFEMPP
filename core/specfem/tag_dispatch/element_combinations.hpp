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
 * @brief Validate a `TagValueTuple` of element or coupling tags, independent of
 *        slot order.
 *
 * Validity depends only on the *physical* tags present — dimension, medium,
 * property, attenuation, boundary for elements; connection, interface,
 * boundary, flux for couplings. Each tag is located by its enum *type*, and
 * dispatch keys on *which* physical tag types appear, not on their slot
 * positions. As a result, slot ordering and interleaved non-physical tags
 * (wavefield, mpi) do not change the outcome.
 *
 * @tparam Tuple  A `TagValueTuple` specialisation.
 * @param  t      The tuple to validate.
 * @return `true` if the combination is physically meaningful.
 */
template <typename Tuple> constexpr bool is_valid(const Tuple &t) {
  using D = specfem::element::dimension_tag;
  using M = specfem::element::medium_tag;
  using P = specfem::element::property_tag;
  using A = specfem::element::attenuation_tag;
  using B = specfem::element::boundary_tag;
  using C = specfem::element_connections::type;

  if constexpr (Tuple::template contains<C>()) {
    // Interface / coupling combinations:
    // (dimension, connection, interface[, boundary[, flux scheme]]).
    using I = specfem::element_coupling::interface_tag;
    using F = specfem::element_coupling::flux_scheme_tag;
    const auto d = t.template get_by_type<D>();
    const auto c = t.template get_by_type<C>();
    const auto i = t.template get_by_type<I>();
    if constexpr (Tuple::template contains<F>())
      return is_valid_edge_and_flux_scheme({ d, c, i,
                                             t.template get_by_type<B>(),
                                             t.template get_by_type<F>() });
    else if constexpr (Tuple::template contains<B>())
      return is_valid_edge({ d, c, i, t.template get_by_type<B>() });
    else
      return is_valid_interface_system({ d, c, i });

  } else if constexpr (Tuple::template contains<D>() &&
                       Tuple::template contains<M>()) {
    // Element combinations: dispatch on which material tags are present.
    const auto d = t.template get_by_type<D>();
    const auto m = t.template get_by_type<M>();
    constexpr bool has_p = Tuple::template contains<P>();
    constexpr bool has_a = Tuple::template contains<A>();
    constexpr bool has_b = Tuple::template contains<B>();

    if constexpr (has_p && has_a && has_b)
      return is_valid_full_combo({ d, m, t.template get_by_type<P>(),
                                   t.template get_by_type<A>(),
                                   t.template get_by_type<B>() });
    else if constexpr (has_p && has_b)
      return is_valid_boundary_combo(
          { d, m, t.template get_by_type<P>(), t.template get_by_type<B>() });
    else if constexpr (has_p && has_a)
      return is_valid_material_combo(
          { d, m, t.template get_by_type<P>(), t.template get_by_type<A>() });
    else if constexpr (has_p)
      return is_valid_property_combo({ d, m, t.template get_by_type<P>() });
    else
      return is_valid_medium_combo({ d, m });

  } else {
    // No connection tag and not a (dimension, medium) element combo — e.g. a
    // wavefield × medium buffer key. No physical constraint applies.
    return true;
  }
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
