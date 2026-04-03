#pragma once

#include "specfem/tags.hpp"
#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

namespace specfem::tag_dispatch::impl {
// ── TagValueTuple<TagTypes...>
// ──────────────────────────────────────────────── Stores a fixed pack of enum
// tag values with constexpr access and assignment. Uses recursive inheritance
// (not std::tuple) so operator= is constexpr in C++17.

template <std::size_t I, typename T> struct TagValueHolder {
  T v{};
  constexpr TagValueHolder() = default;
  constexpr explicit TagValueHolder(T val) : v{ val } {}
};

template <typename IndexSeq, typename... TagTypes> struct TagValueTupleBase;

template <std::size_t... Is, typename... TagTypes>
struct TagValueTupleBase<std::index_sequence<Is...>, TagTypes...>
    : TagValueHolder<Is, TagTypes>... {
  static constexpr std::size_t arity = sizeof...(TagTypes);

  constexpr TagValueTupleBase() = default;
  constexpr TagValueTupleBase(TagTypes... vs)
      : TagValueHolder<Is, TagTypes>{ vs }... {}

  template <std::size_t I> constexpr auto get() const {
    using T = typename std::tuple_element<I, std::tuple<TagTypes...> >::type;
    return static_cast<const TagValueHolder<I, T> &>(*this).v;
  }

  constexpr bool operator==(const TagValueTupleBase &o) const {
    return (... &&
            (static_cast<const TagValueHolder<Is, TagTypes> &>(*this).v ==
             static_cast<const TagValueHolder<Is, TagTypes> &>(o).v));
  }

  std::string name() const {
    std::string s;
    bool first = true;
    ((s += (first ? (first = false, std::string{}) : std::string{ "_" }) +
           specfem::element::to_string(get<Is>())),
     ...);
    return s;
  }
};

template <typename... TagTypes>
using TagValueTuple =
    impl::TagValueTupleBase<std::make_index_sequence<sizeof...(TagTypes)>,
                            TagTypes...>;

// ── Named tuple aliases
// ───────────────────────────────────────────────────────

using MediumTagTuple = TagValueTuple<specfem::element::dimension_tag,
                                     specfem::element::medium_tag>;
using PropertyComboTuple =
    TagValueTuple<specfem::element::dimension_tag, specfem::element::medium_tag,
                  specfem::element::property_tag>;
using MaterialTagTuple =
    TagValueTuple<specfem::element::dimension_tag, specfem::element::medium_tag,
                  specfem::element::property_tag,
                  specfem::element::attenuation_tag>;
using BoundaryComboTuple =
    TagValueTuple<specfem::element::dimension_tag, specfem::element::medium_tag,
                  specfem::element::property_tag,
                  specfem::element::boundary_tag>;
using ElementTagTuple =
    TagValueTuple<specfem::element::dimension_tag, specfem::element::medium_tag,
                  specfem::element::property_tag,
                  specfem::element::attenuation_tag,
                  specfem::element::boundary_tag>;

// ── Rule-based validity predicates
// ───────────────────────────────────────────── Source of truth — no hardcoded
// arrays, just composable rules.

constexpr bool is_valid_medium_combo(MediumTagTuple t) {
  using D = specfem::element::dimension_tag;
  using M = specfem::element::medium_tag;
  auto d = t.get<0>();
  auto m = t.get<1>();
  if (d == D::dim2)
    return m == M::elastic_psv || m == M::elastic_sh || m == M::elastic_psv_t ||
           m == M::acoustic || m == M::poroelastic ||
           m == M::electromagnetic_te;
  if (d == D::dim3)
    return m == M::elastic || m == M::acoustic || m == M::elastic_spin;
  return false;
}

constexpr bool is_valid_property_combo(PropertyComboTuple t) {
  using M = specfem::element::medium_tag;
  using P = specfem::element::property_tag;
  auto d = t.get<0>();
  auto m = t.get<1>();
  auto p = t.get<2>();

  if (!is_valid_medium_combo({ d, m }))
    return false;

  switch (m) {
  case M::elastic_psv:
  case M::elastic_sh:
    return p == P::isotropic || p == P::anisotropic;
  case M::elastic: // dim3 elastic: isotropic only
    return p == P::isotropic;
  case M::elastic_psv_t:
  case M::elastic_spin: // Cosserat elastic: isotropic_cosserat
    return p == P::isotropic_cosserat;
  case M::acoustic:
  case M::poroelastic:
  case M::electromagnetic_te:
    return p == P::isotropic;
  case M::electromagnetic:
    return false; // not a material-level medium
  }
  return false;
}

constexpr bool is_valid_material_combo(MaterialTagTuple t) {
  using M = specfem::element::medium_tag;
  using A = specfem::element::attenuation_tag;
  auto d = t.get<0>();
  auto m = t.get<1>();
  auto p = t.get<2>();
  auto a = t.get<3>();

  if (!is_valid_property_combo({ d, m, p }))
    return false;

  switch (m) {
  case M::elastic_psv_t:
  case M::elastic_spin:       // Cosserat elastic: isotropic_cosserat × none
  case M::electromagnetic_te: // isotropic × none
    return a == A::none;
  default:
    return true; // none or constant_isotropic both valid
  }
}

constexpr bool is_valid_boundary_combo(BoundaryComboTuple t) {
  using D = specfem::element::dimension_tag;
  using M = specfem::element::medium_tag;
  using B = specfem::element::boundary_tag;
  auto d = t.get<0>();
  auto m = t.get<1>();
  auto p = t.get<2>();
  auto b = t.get<3>();

  if (!is_valid_property_combo({ d, m, p }))
    return false;

  // dim3: boundary must be none
  if (d == D::dim3)
    return b == B::none;

  switch (m) {
  case M::acoustic:
    return true; // all boundary types allowed
  case M::electromagnetic_te:
    return b == B::none;
  default:
    return b == B::none || b == B::stacey;
  }
}

constexpr bool is_valid_full_combo(ElementTagTuple t) {
  auto d = t.get<0>();
  auto m = t.get<1>();
  auto p = t.get<2>();
  auto a = t.get<3>();
  auto b = t.get<4>();

  return is_valid_material_combo({ d, m, p, a }) &&
         is_valid_boundary_combo({ d, m, p, b });
}

} // namespace specfem::tag_dispatch::impl
