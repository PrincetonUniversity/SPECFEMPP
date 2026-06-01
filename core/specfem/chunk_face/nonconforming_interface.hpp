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

/**
 * @brief Template accessor for intersection normal vectors
 *
 * Provides chunk-based access to unit normal vectors at intersection
 * quadrature points in nonconforming interfaces.
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam InterfaceTag Interface medium type
 * @tparam BoundaryTag Boundary condition tag
 * @tparam NumberElements Number of edges in chunk
 * @tparam NQuadIntersection Quadrature points on intersection
 * @tparam MemorySpace Kokkos memory space
 * @tparam MemoryTraits Kokkos memory traits
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection,
          typename... KokkosViewArguments>
struct intersection_normal;

template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection,
          typename... KokkosViewArguments>
struct intersection_normal<specfem::element::dimension_tag::dim3, InterfaceTag,
                           BoundaryTag, FluxSchemeTag, NumberElements,
                           NQuadIntersection, KokkosViewArguments...>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::chunk_face,
          specfem::data_access::DataClassType::intersection_normal,
          specfem::element::dimension_tag::dim2, false> {

public:
  /// Spatial dimension (2D for this specialization)
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
  /// Interface medium type tag
  static constexpr auto interface_tag = InterfaceTag;
  /// Boundary condition tag
  static constexpr auto boundary_tag = BoundaryTag;
  /// Flux scheme tag
  static constexpr auto flux_scheme_tag = FluxSchemeTag;
  /// Connection type for nonconforming interfaces
  static constexpr auto connection_tag =
      specfem::element_connections::type::nonconforming;
  /// Number of edges in chunk
  static constexpr int chunk_size = NumberElements;
  /// Number of quadrature points on intersection
  static constexpr int n_quad_intersection = NQuadIntersection;
  /// View type for storing 2D normal vector components
  using IntersectionNormalViewType = specfem::datatype::VectorChunkFaceViewType<
      type_real, dimension_tag, chunk_size, n_quad_intersection, 3,
      false /*UseSIMD=*/, KokkosViewArguments...>;

private:
  /// Underlying view storing normal vector data
  IntersectionNormalViewType data_;

public:
  /**
   * @brief Construct from compatible view type
   * @tparam U Compatible view type
   * @param intersection_normal View containing normal vector data
   */
  template <
      typename U = IntersectionNormalViewType,
      typename std::enable_if_t<
          std::is_convertible<IntersectionNormalViewType, U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION intersection_normal(const U &intersection_normal)
      : data_(intersection_normal) {}

  /**
   * @brief Default constructor
   */
  KOKKOS_INLINE_FUNCTION
  intersection_normal() = default;

  /**
   * @brief Construct from team scratch memory
   * @tparam MemberType Team member type
   * @param team Team member for scratch allocation
   */
  template <typename MemberType, typename U = IntersectionNormalViewType,
            typename std::enable_if_t<U::memory_traits::is_unmanaged == true,
                                      int> = 0>
  KOKKOS_INLINE_FUNCTION intersection_normal(const MemberType &team)
      : data_(team.team_scratch(0)) {}

  /**
   * @brief Get shared memory size requirement
   * @return Size in bytes needed for scratch memory
   */
  constexpr static int shmem_size() {
    return IntersectionNormalViewType::shmem_size();
  }

  /**
   * @brief Access normal vector component
   * @tparam Indices Index types for multi-dimensional access
   * @param indices Element indices (edge, intersection_quad, component)
   * @return Reference to normal vector component
   */
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto &operator()(Indices... indices) const {
    return data_(indices...);
  }
};

/**
 * @brief Variadic template for packing multiple nonconforming interface
 * accessors
 *
 * Combines multiple accessor types (transfer functions, intersection factors,
 * normals) into a single accessor for coordinated access to nonconforming
 * interface data.
 *
 * @tparam Accessors Variadic list of accessor types to pack together
 */
template <typename... Accessors>
struct NonconformingAccessorPack
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::chunk_face,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::element::dimension_tag::dim2, false>,
      public Accessors... {

  /// Spatial dimension inherited from first accessor
  constexpr static auto dimension_tag =
      std::tuple_element_t<0, std::tuple<Accessors...>>::dimension_tag;
  /// Interface medium type inherited from first accessor
  constexpr static auto interface_tag =
      std::tuple_element_t<0, std::tuple<Accessors...>>::interface_tag;
  /// Boundary condition tag inherited from first accessor
  constexpr static auto boundary_tag =
      std::tuple_element_t<0, std::tuple<Accessors...>>::boundary_tag;
  /// Flux scheme tag inherited from first accessor
  constexpr static auto flux_scheme_tag =
      std::tuple_element_t<0, std::tuple<Accessors...>>::flux_scheme_tag;
  /// Number of packed accessor types
  constexpr static size_t n_accessors = sizeof...(Accessors);
  /// Tuple type containing all packed accessors
  using packed_accessors = std::tuple<Accessors...>;
  /// Connection type for nonconforming interfaces
  constexpr static auto connection_tag =
      specfem::element_connections::type::nonconforming;

  /// Data class type for nonconforming interface data
  constexpr static auto data_class =
      specfem::data_access::DataClassType::nonconforming_interface;

  static_assert(
      (std::is_same_v<std::integral_constant<specfem::element::dimension_tag,
                                             Accessors::dimension_tag>,
                      std::integral_constant<specfem::element::dimension_tag,
                                             dimension_tag>> &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "dimension_tag");

  static_assert(
      (std::is_same_v<
           std::integral_constant<specfem::element_coupling::interface_tag,
                                  Accessors::interface_tag>,
           std::integral_constant<specfem::element_coupling::interface_tag,
                                  interface_tag>> &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "interface_tag");

  static_assert(
      (std::is_same_v<std::integral_constant<specfem::element::boundary_tag,
                                             Accessors::boundary_tag>,
                      std::integral_constant<specfem::element::boundary_tag,
                                             boundary_tag>> &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "boundary_tag");

  static_assert(
      (std::is_same_v<
           std::integral_constant<specfem::element_coupling::flux_scheme_tag,
                                  Accessors::flux_scheme_tag>,
           std::integral_constant<specfem::element_coupling::flux_scheme_tag,
                                  flux_scheme_tag>> &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "flux_scheme_tag");

  /**
   * @brief Default constructor
   */
  KOKKOS_INLINE_FUNCTION NonconformingAccessorPack() = default;

  /**
   * @brief Construct from accessor instances
   * @param accessors Individual accessor instances to pack
   */
  template <typename... AcessorInitializers>
  KOKKOS_INLINE_FUNCTION
  NonconformingAccessorPack(const AcessorInitializers &...accessors)
      : Accessors(accessors)... {};

  /**
   * @brief Deleted function call operator (use accessor-specific access)
   */
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION type_real operator()(Indices... indices) const =
      delete;

  /**
   * @brief Construct from team scratch memory
   * @tparam MemberType Team member type
   * @param team Team member for scratch allocation
   */
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION NonconformingAccessorPack(const MemberType &team)
      : Accessors(team)... {}

  /**
   * @brief Get total shared memory size requirement
   * @return Sum of memory requirements for all packed accessors
   */
  constexpr static int shmem_size() {
    return (Accessors::shmem_size() + ... + 0);
  }
};

} // namespace specfem::chunk_face
