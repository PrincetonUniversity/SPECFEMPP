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
using MaterialTagTuple =
    TagValueTuple<specfem::element::dimension_tag, specfem::element::medium_tag,
                  specfem::element::property_tag,
                  specfem::element::attenuation_tag>;
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
  auto d = t.get<0>();
  // dim2: all media defined; dim3: none yet
  return d == D::dim2;
}

constexpr bool is_valid_material_combo(MaterialTagTuple t) {
  using M = specfem::element::medium_tag;
  using P = specfem::element::property_tag;
  using A = specfem::element::attenuation_tag;
  auto d = t.get<0>();
  auto m = t.get<1>();
  auto p = t.get<2>();
  auto a = t.get<3>();

  if (!is_valid_medium_combo({ d, m }))
    return false;

  switch (m) {
  case M::elastic_psv:
  case M::elastic_sh:
    // isotropic × {none, ci}  or  anisotropic × {none}
    if (p == P::isotropic)
      return true;
    if (p == P::anisotropic)
      return a == A::none;
    return false;
  case M::elastic_psv_t:
    return p == P::isotropic_cosserat && a == A::none;
  case M::acoustic:
    return p == P::isotropic; // {none, ci} both allowed
  case M::poroelastic:
  case M::electromagnetic_te:
    return p == P::isotropic && a == A::none;
  case M::elastic:
  case M::elastic_spin:
  case M::electromagnetic:
    return false; // not a material-level medium
  }
  return false;
}

constexpr bool is_valid_full_combo(ElementTagTuple t) {
  using M = specfem::element::medium_tag;
  using B = specfem::element::boundary_tag;
  auto d = t.get<0>();
  auto m = t.get<1>();
  auto p = t.get<2>();
  auto a = t.get<3>();
  auto b = t.get<4>();

  if (!is_valid_material_combo({ d, m, p, a }))
    return false;

  switch (m) {
  case M::acoustic:
    return true; // all boundary types allowed
  case M::electromagnetic_te:
    return b == B::none;
  default:
    return b == B::none || b == B::stacey;
  }
}

} // namespace specfem::tag_dispatch::impl
