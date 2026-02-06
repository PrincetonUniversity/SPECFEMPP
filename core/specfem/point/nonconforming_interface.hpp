#pragma once

#include "specfem/data_access.hpp"

namespace specfem::point {

namespace impl {

/**
 * @struct nonconforming_transfer_function
 * @brief Accessor for non-conforming interface transfer functions.
 *
 * @tparam NQuadIntersection Number of quadrature points in the intersection.
 * @tparam DimensionTag Spatial dimension.
 * @tparam DataClass Data class type.
 * @tparam InterfaceTag Interface type.
 * @tparam BoundaryTag Boundary condition type.
 */
template <int NQuadIntersection, specfem::element::dimension_tag DimensionTag,
          specfem::data_access::DataClassType DataClass,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct nonconforming_transfer_function
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point, DataClass, DimensionTag,
          false> {
public:
  /**
   * @brief Interface type tag.
   */
  static constexpr auto interface_tag = InterfaceTag;

  /**
   * @brief Boundary condition tag.
   */
  static constexpr auto boundary_tag = BoundaryTag;

  /**
   * @brief Number of quadrature points in the intersection.
   */
  static constexpr auto n_quad_intersection = NQuadIntersection;

  /**
   * @brief Connection type tag (nonconforming).
   */
  static constexpr auto connection_tag =
      specfem::element_connections::type::nonconforming;

  /**
   * @brief View type for the transfer function values.
   */
  using TransferViewType =
      specfem::datatype::VectorPointViewType<type_real, NQuadIntersection,
                                             false>;

  /**
   * @brief Transfer function values.
   */
  TransferViewType transfer_function;

  /**
   * @brief Constructor with transfer function values.
   *
   * @param transfer_function The transfer function values.
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_transfer_function(const TransferViewType &transfer_function)
      : transfer_function(transfer_function) {}

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_transfer_function() = default;

  /**
   * @brief Accessor operator (const).
   *
   * @tparam Indices Index types.
   * @param indices Indices to access the transfer function.
   * @return Const reference to the value.
   */
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION const auto &operator()(Indices... indices) const {
    return transfer_function(indices...);
  }

  /**
   * @brief Accessor operator (mutable).
   *
   * @tparam Indices Index types.
   * @param indices Indices to access the transfer function.
   * @return Reference to the value.
   */
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto &operator()(Indices... indices) {
    return transfer_function(indices...);
  }
};

} // namespace impl

/**
 * @brief Type alias for self-side transfer function.
 */
template <int NQuadIntersection, specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
using transfer_function_self = impl::nonconforming_transfer_function<
    NQuadIntersection, DimensionTag,
    specfem::data_access::DataClassType::transfer_function_self, InterfaceTag,
    BoundaryTag>;

/**
 * @brief Type alias for coupled-side transfer function.
 */
template <int NQuadIntersection, specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
using transfer_function_coupled = impl::nonconforming_transfer_function<
    NQuadIntersection, DimensionTag,
    specfem::data_access::DataClassType::transfer_function_coupled,
    InterfaceTag, BoundaryTag>;

/**
 * @struct NonconformingAccessorPack
 * @brief Packs multiple accessors for non-conforming interfaces.
 *
 * This struct aggregates multiple accessors (e.g., transfer functions) into a
 * single object that satisfies the accessor interface.
 *
 * @tparam Accessors Variadic list of accessor types.
 */
template <typename... Accessors>
struct NonconformingAccessorPack
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::element::dimension_tag::dim2, false>,
      public Accessors... {

private:
  /**
   * @brief Base accessor type alias.
   */
  using accessor_base = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::nonconforming_interface,
      specfem::element::dimension_tag::dim2, false>;

public:
  /**
   * @brief Interface type tag derived from the first accessor.
   */
  constexpr static auto interface_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::interface_tag;

  /**
   * @brief Boundary condition tag derived from the first accessor.
   */
  constexpr static auto boundary_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::boundary_tag;

  /**
   * @brief Number of packed accessors.
   */
  constexpr static size_t n_accessors = sizeof...(Accessors);

  /**
   * @brief Tuple type of packed accessors.
   */
  using packed_accessors = std::tuple<Accessors...>;

  /**
   * @brief Connection type tag (nonconforming).
   */
  constexpr static auto connection_tag =
      specfem::element_connections::type::nonconforming;

  /**
   * @brief Dimension tag.
   */
  constexpr static auto dimension_tag = accessor_base::dimension_tag;

  /**
   * @brief Data class type.
   */
  constexpr static auto data_class = accessor_base::data_class;

  /**
   * @brief Accessor type.
   */
  constexpr static auto accessor_type = accessor_base::accessor_type;

  /**
   * @brief SIMD usage flag.
   */
  constexpr static bool using_simd = accessor_base::using_simd;

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

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  NonconformingAccessorPack() = default;

  /**
   * @brief Constructor with accessors.
   *
   * @param accessors The accessors to pack.
   */
  KOKKOS_INLINE_FUNCTION
  NonconformingAccessorPack(const Accessors &...accessors)
      : Accessors(accessors)... {};

  /**
   * @brief Deleted operator() to prevent direct access.
   */
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION type_real operator()(Indices... indices) const =
      delete;
};

/**
 * @brief Type alias for a pack containing the coupled transfer function.
 */
template <int NQuadIntersection, specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
using transfer_function_pack =
    NonconformingAccessorPack<transfer_function_coupled<
        NQuadIntersection, DimensionTag, InterfaceTag, BoundaryTag> >;

} // namespace specfem::point
