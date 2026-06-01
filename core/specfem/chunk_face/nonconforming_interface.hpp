#pragma once

#include "specfem/data_access/accessor.hpp"
#include "specfem/datatype/chunk_face_view.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element_connections/tags.hpp"
#include "specfem/element_coupling/tags.hpp"

#include <Kokkos_Core.hpp>

namespace specfem::chunk_face {

/**
 * @brief Template accessor for face node location in coupled local coordinates.
 *
 * @tparam DimensionTag
 * @tparam NumberElements Chunk size
 * @tparam NQuadElement assembly NGLL
 * @tparam InterfaceTag
 * @tparam BoundaryTag
 * @tparam FluxSchemeTag
 * @tparam MemorySpace
 * @tparam MemoryTraits
 */
template <specfem::element::dimension_tag DimensionTag, int NumberElements,
          int NQuadElement,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          typename... KokkosViewArguments>
struct coupled_coordinates;

template <int NumberElements, int NQuadElement,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          typename... KokkosViewArguments>
struct coupled_coordinates<specfem::element::dimension_tag::dim3,
                           NumberElements, NQuadElement, InterfaceTag,
                           BoundaryTag, FluxSchemeTag, KokkosViewArguments...>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::coupled_coordinates,
          specfem::element::dimension_tag::dim3, false> {

public:
  /// Spatial dimension (2D for this specialization)
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
  /// Number of edges in chunk
  static constexpr int chunk_size = NumberElements;
  /// Number of quadrature points on element edge
  static constexpr int n_quad_element = NQuadElement;
  /// Interface medium type tag
  static constexpr auto interface_tag = InterfaceTag;
  /// Boundary condition tag
  static constexpr auto boundary_tag = BoundaryTag;
  /// Flux scheme tag
  static constexpr auto flux_scheme_tag = FluxSchemeTag;
  /// Connection type for nonconforming interfaces
  static constexpr auto connection_tag =
      specfem::element_connections::type::nonconforming;
  using ViewType = specfem::datatype::VectorChunkFaceViewType<
      type_real, specfem::element::dimension_tag::dim3, NumberElements,
      NQuadElement, specfem::element::dimension<dimension_tag>::dim - 1, false,
      KokkosViewArguments...>; ///< Underlying view storing node coordinates

private:
  /// Underlying view storing node coordinates
  ViewType data_;

public:
  /**
   * @brief Construct from compatible view type
   * @tparam U Compatible view type
   */
  template <typename U = ViewType>
  KOKKOS_INLINE_FUNCTION coupled_coordinates(const U &coupled_coordinates)
      : data_(coupled_coordinates) {}

  /**
   * @brief Default constructor
   */
  KOKKOS_INLINE_FUNCTION
  coupled_coordinates() = default;

  /**
   * @brief Construct from team scratch memory
   * @tparam MemberType Team member type
   * @param team Team member for scratch allocation
   */
  template <typename MemberType, typename U = ViewType,
            typename std::enable_if_t<U::memory_traits::is_unmanaged == true,
                                      int> = 0>
  KOKKOS_INLINE_FUNCTION coupled_coordinates(const MemberType &team)
      : data_(team.team_scratch(0)) {}

  /**
   * @brief Get shared memory size requirement
   * @return Size in bytes needed for scratch memory
   */
  constexpr static int shmem_size() { return ViewType::shmem_size(); }

  /**
   * @brief Access transfer function matrix element
   * @tparam Indices Index types for multi-dimensional access
   * @param indices Element indices (edge, intersection_quad, edge_quad)
   * @return Reference to matrix element
   */
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto &operator()(Indices... indices) const {
    return data_(indices...);
  }
};

} // namespace specfem::chunk_face
