#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/element_connections/tags.hpp"
#include "specfem/element_connections/to_string.hpp"
#include "specfem/element_coupling/tags.hpp"
#include "specfem/element_coupling/to_string.hpp"
#include "specfem/simulation.hpp"

#include <string>
#include <type_traits>

namespace specfem::tags {

/**
 * @brief Template specialization for storing compile-time tag values.
 *
 * Base template for storing typed enumeration values as static constexpr
 * members. Specialized for each tag type (dimension, medium, property,
 * boundary, interface, etc.) to enable compile-time tag dispatch and type-safe
 * element classification.
 *
 * @tparam TagType Enumeration type (e.g., medium_tag, boundary_tag)
 * @tparam Value Specific enum value to store
 */
template <typename TagType, TagType Value> struct TagMember;

/**
 * @brief Dimension tag specialization for 2D/3D simulation dispatch.
 */
template <specfem::element::dimension_tag DimensionTag>
struct TagMember<specfem::element::dimension_tag, DimensionTag> {
  static constexpr specfem::element::dimension_tag dimension_tag = DimensionTag;
};

/**
 * @brief Medium tag specialization for physics classification (elastic,
 * acoustic, etc.).
 */
template <specfem::element::medium_tag MediumTag>
struct TagMember<specfem::element::medium_tag, MediumTag> {
  static constexpr specfem::element::medium_tag medium_tag = MediumTag;
};

/**
 * @brief Property tag specialization for material anisotropy (isotropic,
 * anisotropic, etc.).
 */
template <specfem::element::property_tag PropertyTag>
struct TagMember<specfem::element::property_tag, PropertyTag> {
  static constexpr specfem::element::property_tag property_tag = PropertyTag;
};

/**
 * @brief Boundary tag specialization for boundary conditions (free surface,
 * Stacey, etc.).
 */
template <specfem::element::boundary_tag BoundaryTag>
struct TagMember<specfem::element::boundary_tag, BoundaryTag> {
  static constexpr specfem::element::boundary_tag boundary_tag = BoundaryTag;
};

/**
 * @brief Interface tag specialization for element-to-element coupling types.
 */
template <specfem::element_coupling::interface_tag InterfaceTag>
struct TagMember<specfem::element_coupling::interface_tag, InterfaceTag> {
  static constexpr specfem::element_coupling::interface_tag interface_tag =
      InterfaceTag;
};

/**
 * @brief Flux scheme tag specialization for numerical flux calculations.
 */
template <specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
struct TagMember<specfem::element_coupling::flux_scheme_tag, FluxSchemeTag> {
  static constexpr specfem::element_coupling::flux_scheme_tag flux_scheme_tag =
      FluxSchemeTag;
};

/**
 * @brief Connection tag specialization for mesh connectivity types.
 */
template <specfem::element_connections::type ConnectionTag>
struct TagMember<specfem::element_connections::type, ConnectionTag> {
  static constexpr specfem::element_connections::type connection_tag =
      ConnectionTag;
};

/**
 * @brief Wavefield tag specialization for simulation field types (forward,
 * adjoint, etc.).
 */
template <specfem::simulation::field_type WavefieldTag>
struct TagMember<specfem::simulation::field_type, WavefieldTag> {
  static constexpr specfem::simulation::field_type wavefield_tag = WavefieldTag;
};

/**
 * @brief Attenuation tag specialization for attenuation model types
 * (none, constant_isotropic, etc.).
 */
template <specfem::element::attenuation_tag AttenuationTag>
struct TagMember<specfem::element::attenuation_tag, AttenuationTag> {
  static constexpr specfem::element::attenuation_tag attenuation_tag =
      AttenuationTag;
};

/**
 * @brief SIMD tag specialization for point field types (enabled, disabled).
 */
template <bool UseSIMD> struct TagMember<bool, UseSIMD> {
  static constexpr bool using_simd = UseSIMD;
};

template <typename A, typename B> struct tag_equals {
  static constexpr bool eval(const A &, const B &) { return false; }
};

template <typename A> struct tag_equals<A, A> {
  static constexpr bool eval(const A &lhs, const A &rhs) { return lhs == rhs; }
};

/**
 * @brief Variadic template for combining multiple compile-time tags.
 *
 * Inherits from multiple TagMember specializations to aggregate different
 * enumeration tags into a single type. Enables compile-time element
 * classification with multiple attributes (dimension, medium, boundary, etc.)
 * for template dispatch.
 *
 * @tparam TagMembers Variadic pack of enumeration values to combine
 *
 * @code
 * using Element2D = Tags<specfem::element::dimension_tag::dim2,
 *                        specfem::element::medium_tag::elastic,
 *                        specfem::element::boundary_tag::stacey>;
 * static_assert(Element2D::dimension_tag ==
 * specfem::element::dimension_tag::dim2);
 * @endcode
 */
template <auto... TagMembers>
struct Tags : TagMember<decltype(TagMembers), TagMembers>... {
private:
  template <typename QueryTagType>
  static constexpr bool has_tag(const QueryTagType query_tag) {
    return (
        specfem::tags::tag_equals<
            decltype(TagMembers),
            std::remove_cvref_t<QueryTagType> >::eval(TagMembers, query_tag) ||
        ...);
  }

public:
  /**
   * @brief Compile-time tag expansion: appends a single tag value to this Tags
   * type.
   *
   * @tparam NewTag The compile-time tag value to append.
   *
   * Example:
   * @code
   * using Combined = ElementTags::expand<wavefield_tag_value>;
   * // Combined has all members of ElementTags plus wavefield_tag
   * @endcode
   */
  template <auto NewTag> using expand = Tags<TagMembers..., NewTag>;

  /**
   * @brief Runtime subset check for one or more tag values.
   *
   * Returns true if every provided runtime tag exists in this `Tags` type.
   */
  template <typename... QueryTagTypes>
  constexpr bool has(const QueryTagTypes... query_tags) const {
    return (Tags::has_tag(query_tags) && ...);
  }

  static std::string name() {
    std::string s;
    bool first = true;
    ((s += (first ? (first = false, std::string{}) : std::string{ "_" }) +
           to_string(TagMembers)),
     ...);
    return s;
  }
};

/**
 * @brief Free alias template for appending a tag to an existing Tags type.
 *
 * Avoids the `typename T::template expand<V>` dependent-name syntax at call
 * sites; use `specfem::tags::expand<TagsType, NewTag>` instead.
 *
 * @tparam TagsType An existing Tags specialization.
 * @tparam NewTag   Compile-time tag value to append.
 */
template <typename TagsType, auto NewTag> struct expand_impl;

template <auto... TagMembers, auto NewTag>
struct expand_impl<Tags<TagMembers...>, NewTag> {
  using type = Tags<TagMembers..., NewTag>;
};

template <typename TagsType, auto NewTag>
using expand = typename expand_impl<TagsType, NewTag>::type;

} // namespace specfem::tags
