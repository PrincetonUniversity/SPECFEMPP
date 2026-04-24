#pragma once

#include "specfem/element_coupling/tags.hpp"
namespace specfem::chunk_edge {

namespace impl {

/**
 * @brief Template accessor for transfer function data at nonconforming
 * interfaces
 *
 * Maps edge functions to intersection functions in mortar methods.
 * Supports both self and coupled interface transfer operations.
 *
 * @tparam DimensionTag Spatial dimension (dim2, dim3)
 * @tparam NumberElements Number of edges in chunk
 * @tparam NQuadIntersection Quadrature points on intersection
 * @tparam NQuadElement Quadrature points on element edge
 * @tparam DataClass Self or coupled transfer data type
 * @tparam InterfaceTag Interface medium type
 * @tparam BoundaryTag Boundary condition tag
 * @tparam MemorySpace Kokkos memory space
 * @tparam MemoryTraits Kokkos memory traits
 */
template <specfem::element::dimension_tag DimensionTag, int NumberElements,
          int NQuadIntersection, int NQuadElement,
          specfem::data_access::DataClassType DataClass,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct transfer_function;

/**
 * @brief 2D specialization of transfer function accessor
 *
 * Provides chunk-based access to transfer function matrices for mapping
 * edge functions to intersection functions in nonconforming interfaces.
 * Used in mortar methods for coupling elements with different mesh sizes.
 */
template <int NumberElements, int NQuadIntersection, int NQuadElement,
          specfem::data_access::DataClassType DataClass,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          typename MemorySpace, typename MemoryTraits>
struct transfer_function<specfem::element::dimension_tag::dim2, NumberElements,
                         NQuadIntersection, NQuadElement, DataClass,
                         InterfaceTag, BoundaryTag, FluxSchemeTag, MemorySpace,
                         MemoryTraits>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::chunk_edge, DataClass,
          specfem::element::dimension_tag::dim2, false> {

public:
  /// Spatial dimension (2D for this specialization)
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;
  /// Number of edges in chunk
  static constexpr int chunk_size = NumberElements;
  /// Number of quadrature points on element edge
  static constexpr int n_quad_element = NQuadElement;
  /// Number of quadrature points on intersection
  static constexpr int n_quad_intersection = NQuadIntersection;
  /// Interface medium type tag
  static constexpr auto interface_tag = InterfaceTag;
  /// Boundary condition tag
  static constexpr auto boundary_tag = BoundaryTag;
  /// Flux scheme tag
  static constexpr auto flux_scheme_tag = FluxSchemeTag;
  /// Connection type for nonconforming interfaces
  static constexpr auto connection_tag =
      specfem::element_connections::type::nonconforming;
  using TransferViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, specfem::element::dimension_tag::dim2, NumberElements,
      NQuadElement, NQuadIntersection, false, MemorySpace,
      MemoryTraits>; ///< Underlying view storing transfer function matrix data

private:
  /// Underlying view storing transfer function matrix data
  TransferViewType data_;

public:
  /**
   * @brief Construct from compatible view type
   * @tparam U Compatible view type
   */
  template <typename U = TransferViewType,
            typename std::enable_if_t<
                std::is_convertible<TransferViewType, U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION transfer_function(const U &transfer_function)
      : data_(transfer_function) {}

  /**
   * @brief Default constructor
   */
  KOKKOS_INLINE_FUNCTION
  transfer_function() = default;

  /**
   * @brief Construct from team scratch memory
   * @tparam MemberType Team member type
   * @param team Team member for scratch allocation
   */
  template <typename MemberType, typename U = TransferViewType,
            typename std::enable_if_t<U::memory_traits::is_unmanaged == true,
                                      int> = 0>
  KOKKOS_INLINE_FUNCTION transfer_function(const MemberType &team)
      : data_(team.team_scratch(0)) {}

  /**
   * @brief Get shared memory size requirement
   * @return Size in bytes needed for scratch memory
   */
  constexpr static int shmem_size() { return TransferViewType::shmem_size(); }

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

} // namespace impl

/**
 * @brief Type alias for self-coupling transfer function
 *
 * Transfer function for mapping edge functions to intersection on the same
 * element.
 *
 * @tparam DimensionTag Spatial dimension (dim2, dim3)
 * @tparam InterfaceTag Interface medium type
 * @tparam BoundaryTag Boundary condition tag
 * @tparam NumberElements Number of edges in chunk
 * @tparam NQuadIntersection Quadrature points on intersection
 * @tparam NQuadElement Quadrature points on element edge
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection, int NQuadElement>
using transfer_function_self = impl::transfer_function<
    DimensionTag, NumberElements, NQuadIntersection, NQuadElement,
    specfem::data_access::DataClassType::transfer_function_self, InterfaceTag,
    BoundaryTag, FluxSchemeTag>;

/**
 * @brief Type alias for coupled transfer function
 *
 * Transfer function for mapping edge functions to intersection from coupled
 * element.
 *
 * @tparam DimensionTag Spatial dimension (dim2, dim3)
 * @tparam InterfaceTag Interface medium type
 * @tparam BoundaryTag Boundary condition tag
 * @tparam NumberElements Number of edges in chunk
 * @tparam NQuadIntersection Quadrature points on intersection
 * @tparam NQuadElement Quadrature points on element edge
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection, int NQuadElement>
using transfer_function_coupled = impl::transfer_function<
    DimensionTag, NumberElements, NQuadIntersection, NQuadElement,
    specfem::data_access::DataClassType::transfer_function_coupled,
    InterfaceTag, BoundaryTag, FluxSchemeTag>;

/**
 * @brief Template accessor for intersection scaling factors
 *
 * Provides chunk-based access to geometric scaling factors applied
 * at intersection quadrature points in nonconforming interfaces.
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
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct intersection_factor;

/**
 * @brief 2D specialization of intersection factor accessor
 *
 * Stores geometric scaling factors for intersection quadrature points.
 */
template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection, typename MemorySpace,
          typename MemoryTraits>
struct intersection_factor<specfem::element::dimension_tag::dim2, InterfaceTag,
                           BoundaryTag, FluxSchemeTag, NumberElements,
                           NQuadIntersection, MemorySpace, MemoryTraits>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::intersection_factor,
          specfem::element::dimension_tag::dim2, false> {

public:
  /// Spatial dimension (2D for this specialization)
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;
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
  /// View type for storing intersection scaling factors
  using IntersectionFactorViewType =
      Kokkos::View<type_real[NumberElements][NQuadIntersection], MemorySpace,
                   MemoryTraits>;

private:
  /// Underlying view storing intersection factor data
  IntersectionFactorViewType data_;

public:
  /**
   * @brief Construct from compatible view type
   * @tparam U Compatible view type
   * @param intersection_factor View containing intersection factor data
   */
  template <
      typename U = IntersectionFactorViewType,
      typename std::enable_if_t<
          std::is_convertible<IntersectionFactorViewType, U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION intersection_factor(const U &intersection_factor)
      : data_(intersection_factor) {}

  /**
   * @brief Default constructor
   */
  KOKKOS_INLINE_FUNCTION
  intersection_factor() = default;

  /**
   * @brief Construct from team scratch memory
   * @tparam MemberType Team member type
   * @param team Team member for scratch allocation
   */
  template <typename MemberType, typename U = IntersectionFactorViewType,
            typename std::enable_if_t<U::memory_traits::is_unmanaged == true,
                                      int> = 0>
  KOKKOS_INLINE_FUNCTION intersection_factor(const MemberType &team)
      : data_(team.team_scratch(0)) {}

  /**
   * @brief Get shared memory size requirement
   * @return Size in bytes needed for scratch memory
   */
  constexpr static int shmem_size() {
    return IntersectionFactorViewType::shmem_size();
  }

  /**
   * @brief Access intersection factor element
   * @tparam Indices Index types for multi-dimensional access
   * @param indices Element indices (edge, intersection_quad)
   * @return Reference to factor value
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
          typename MemorySpace =
              Kokkos::DefaultExecutionSpace::scratch_memory_space,
          typename MemoryTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged> >
struct intersection_normal;

/**
 * @brief 2D specialization of intersection normal accessor
 *
 * Stores 2D unit normal vectors at intersection quadrature points.
 */
template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection, typename MemorySpace,
          typename MemoryTraits>
struct intersection_normal<specfem::element::dimension_tag::dim2, InterfaceTag,
                           BoundaryTag, FluxSchemeTag, NumberElements,
                           NQuadIntersection, MemorySpace, MemoryTraits>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::intersection_normal,
          specfem::element::dimension_tag::dim2, false> {

public:
  /// Spatial dimension (2D for this specialization)
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;
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
  using IntersectionNormalViewType =
      Kokkos::View<type_real[NumberElements][NQuadIntersection][2], MemorySpace,
                   MemoryTraits>;

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
          specfem::datatype::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::element::dimension_tag::dim2, false>,
      public Accessors... {

  /// Spatial dimension inherited from first accessor
  constexpr static auto dimension_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::dimension_tag;
  /// Interface medium type inherited from first accessor
  constexpr static auto interface_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::interface_tag;
  /// Boundary condition tag inherited from first accessor
  constexpr static auto boundary_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::boundary_tag;
  /// Flux scheme tag inherited from first accessor
  constexpr static auto flux_scheme_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::flux_scheme_tag;
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
                                             dimension_tag> > &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "dimension_tag");

  static_assert(
      (std::is_same_v<
           std::integral_constant<specfem::element_coupling::interface_tag,
                                  Accessors::interface_tag>,
           std::integral_constant<specfem::element_coupling::interface_tag,
                                  interface_tag> > &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "interface_tag");

  static_assert(
      (std::is_same_v<std::integral_constant<specfem::element::boundary_tag,
                                             Accessors::boundary_tag>,
                      std::integral_constant<specfem::element::boundary_tag,
                                             boundary_tag> > &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "boundary_tag");

  static_assert(
      (std::is_same_v<
           std::integral_constant<specfem::element_coupling::flux_scheme_tag,
                                  Accessors::flux_scheme_tag>,
           std::integral_constant<specfem::element_coupling::flux_scheme_tag,
                                  flux_scheme_tag> > &&
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
  KOKKOS_INLINE_FUNCTION
  NonconformingAccessorPack(const Accessors &...accessors)
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
          int NumberElements, int NQuadIntersection, int NQuadElement>
using coupling_terms_pack = NonconformingAccessorPack<
    transfer_function_coupled<DimensionTag, InterfaceTag, BoundaryTag,
                              FluxSchemeTag, NumberElements, NQuadIntersection,
                              NQuadElement>,
    intersection_normal<DimensionTag, InterfaceTag, BoundaryTag, FluxSchemeTag,
                        NumberElements, NQuadIntersection> >;

/**
 * @brief Type alias for integral data accessor pack
 *
 * Provides access to intersection scaling factors for numerical integration
 * at nonconforming interfaces.
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          int NumberElements, int NQuadIntersection>
using integral_data_pack = NonconformingAccessorPack<
    intersection_factor<DimensionTag, InterfaceTag, BoundaryTag, FluxSchemeTag,
                        NumberElements, NQuadIntersection> >;

} // namespace specfem::chunk_edge
