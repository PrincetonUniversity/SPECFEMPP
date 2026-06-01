#pragma once

#include "specfem/chunk_edge/nonconforming_interface.hpp"
#include "specfem/chunk_face/nonconforming_interface.hpp"

#include <tuple>
#include <type_traits>

namespace specfem::element_coupling::accessor {

namespace impl {

// ===========================================================================
// impl for overloading type alias

template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection, int NQuadElement,
          typename KokkosViewArgsTuple = std::tuple<>, typename = void>
struct coupling_terms_pack;

template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection,
          typename KokkosViewArgsTuple = std::tuple<>, typename = void>
struct intersection_factor;

// ===========================================================================
// coupling_terms_pack
// ===========================================================================

template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection, int NQuadElement,
          typename... KokkosViewArgs>
struct coupling_terms_pack<
    DimensionTag, InterfaceTag, BoundaryTag, FluxSchemeTag, NumberElements,
    NQuadIntersection, NQuadElement, std::tuple<KokkosViewArgs...>,
    std::enable_if_t<DimensionTag == specfem::element::dimension_tag::dim2>> {
  static_assert(sizeof...(KokkosViewArgs) == 0,
                "This coupling_terms_pack does not have Kokkos-view-argument "
                "passthrough implemented!");
  using type = specfem::chunk_edge::coupling_terms_pack<
      DimensionTag, InterfaceTag, BoundaryTag, FluxSchemeTag, NumberElements,
      NQuadIntersection, NQuadElement>;
};

template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection, int NQuadElement,
          typename... KokkosViewArgs>
struct coupling_terms_pack<
    DimensionTag, InterfaceTag, BoundaryTag, FluxSchemeTag, NumberElements,
    NQuadIntersection, NQuadElement, std::tuple<KokkosViewArgs...>,
    std::enable_if_t<DimensionTag == specfem::element::dimension_tag::dim3>> {
  using type = specfem::chunk_face::NonconformingAccessorPack<
      specfem::chunk_face::coupled_coordinates<
          DimensionTag, NumberElements, NQuadElement, InterfaceTag, BoundaryTag,
          FluxSchemeTag, KokkosViewArgs...>,
      specfem::chunk_face::intersection_normal<
          DimensionTag, InterfaceTag, BoundaryTag, FluxSchemeTag,
          NumberElements, NQuadElement, KokkosViewArgs...>>;
};

// ===========================================================================
// intersection_factor
// ===========================================================================

template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection, typename... KokkosViewArgs>
struct intersection_factor<
    DimensionTag, InterfaceTag, BoundaryTag, FluxSchemeTag, NumberElements,
    NQuadIntersection, std::tuple<KokkosViewArgs...>,
    std::enable_if_t<DimensionTag == specfem::element::dimension_tag::dim2>> {
  static_assert(sizeof...(KokkosViewArgs) == 0,
                "This coupling_terms_pack does not have Kokkos-view-argument "
                "passthrough implemented!");
  using type = specfem::chunk_edge::intersection_factor<
      DimensionTag, InterfaceTag, BoundaryTag, FluxSchemeTag, NumberElements,
      NQuadIntersection>;
};

// ===========================================================================

} // namespace impl

/**
 * @brief Type alias for coupling terms accessor pack
 *
 * Combines transfer function and intersection normal accessors for
 * computing coupling terms in nonconforming interface methods.
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection, int NQuadElement,
          typename... KokkosViewArguments>
using coupling_terms_pack = impl::coupling_terms_pack<
    DimensionTag, InterfaceTag, BoundaryTag, FluxSchemeTag, NumberElements,
    NQuadIntersection, NQuadElement, std::tuple<KokkosViewArguments...>>::type;

template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection,
          typename... KokkosViewArguments>
using intersection_factor =
    impl::intersection_factor<DimensionTag, InterfaceTag, BoundaryTag,
                              FluxSchemeTag, NumberElements, NQuadIntersection,
                              std::tuple<KokkosViewArguments...>>::type;
} // namespace specfem::element_coupling::accessor
