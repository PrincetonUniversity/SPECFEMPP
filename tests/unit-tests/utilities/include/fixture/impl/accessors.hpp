#pragma once

#include "specfem/data_access/accessor.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/element_coupling.hpp"
namespace specfem::test_fixture::impl {

/**
 * @brief Baseline view for a nonconforming data accessor
 * (core/specfem/chunk_edge/nonconforming_interface.hpp)
 *
 * @tparam InterfaceTag
 * @tparam BoundaryTag
 * @tparam DataClassType
 * @tparam Axes The size of the view along each axis.
 */
template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::data_access::DataClassType DataClassType, int... Axes>
struct NonconformingAccessorPatch2D
    : specfem::data_access::Accessor<
          specfem::datatype::AccessorType::chunk_edge, DataClassType,
          specfem::element::dimension_tag::dim2, false /* UseSIMD */> {
public:
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto connection_tag =
      specfem::element_connections::type::nonconforming;
  /// View type for storing intersection scaling factors
private:
  // ======================================================================================
  /**
   * @brief helper to unpack [Axes][...]
   *
   * ViewInternalType<T,Axis1, ..., Axisk>::type = T[Axis1]...[Axisk]
   */
  template <typename LeftType, int... RemainingAxes> struct ViewInternalType;

  // stitch in axis recursively.
  template <typename LeftType, int Axis, int... RemainingAxes>
  struct ViewInternalType<LeftType, Axis, RemainingAxes...> {

    // I would have thought it would be
    // ViewInternalType<LeftType[Axis], RemainingAxes...>::type
    // but this one gave me the correct type:
    using type =
        typename ViewInternalType<LeftType, RemainingAxes...>::type[Axis];
  };

  // base case (recursive termination)
  template <typename LeftType> struct ViewInternalType<LeftType> {
    using type = LeftType;
  };
  // ======================================================================================

public:
  using DataViewType =
      Kokkos::View<typename ViewInternalType<type_real, Axes...>::type,
                   Kokkos::DefaultExecutionSpace::memory_space>;

private:
  /// Underlying view storing transfer function matrix data
  DataViewType data_;

public:
  KOKKOS_INLINE_FUNCTION
  NonconformingAccessorPatch2D() = default;

  NonconformingAccessorPatch2D(const std::string &name) : data_(name) {}

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

  typename DataViewType::HostMirror create_host_mirror() {
    return Kokkos::create_mirror_view(data_);
  }

  void sync_to_device(const typename DataViewType::HostMirror &host_data) {
    Kokkos::deep_copy(data_, host_data);
  }
};

template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadElement, int NQuadIntersection>
struct NonconformingTransferFunctionSelfPatch
    : NonconformingAccessorPatch2D<
          InterfaceTag, BoundaryTag,
          specfem::data_access::DataClassType::transfer_function_self,
          NumberElements, NQuadElement, NQuadIntersection> {
  using NonconformingAccessorPatch2D<
      InterfaceTag, BoundaryTag,
      specfem::data_access::DataClassType::transfer_function_self,
      NumberElements, NQuadElement,
      NQuadIntersection>::NonconformingAccessorPatch2D;
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_element = NQuadElement;
  static constexpr int n_quad_intersection = NQuadIntersection;
};
template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadElement, int NQuadIntersection>
struct NonconformingTransferFunctionCoupledPatch
    : NonconformingAccessorPatch2D<
          InterfaceTag, BoundaryTag,
          specfem::data_access::DataClassType::transfer_function_coupled,
          NumberElements, NQuadElement, NQuadIntersection> {
  using NonconformingAccessorPatch2D<
      InterfaceTag, BoundaryTag,
      specfem::data_access::DataClassType::transfer_function_coupled,
      NumberElements, NQuadElement,
      NQuadIntersection>::NonconformingAccessorPatch2D;
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_element = NQuadElement;
  static constexpr int n_quad_intersection = NQuadIntersection;
};
template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection>
struct NonconformingIntersectionNormalPatch
    : NonconformingAccessorPatch2D<
          InterfaceTag, BoundaryTag,
          specfem::data_access::DataClassType::intersection_normal,
          NumberElements, NQuadIntersection, 2> {
  using NonconformingAccessorPatch2D<
      InterfaceTag, BoundaryTag,
      specfem::data_access::DataClassType::intersection_normal, NumberElements,
      NQuadIntersection, 2>::NonconformingAccessorPatch2D;
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_intersection = NQuadIntersection;
};
template <specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection>
struct NonconformingIntersectionFactorPatch
    : NonconformingAccessorPatch2D<
          InterfaceTag, BoundaryTag,
          specfem::data_access::DataClassType::intersection_factor,
          NumberElements, NQuadIntersection> {
  using NonconformingAccessorPatch2D<
      InterfaceTag, BoundaryTag,
      specfem::data_access::DataClassType::intersection_factor, NumberElements,
      NQuadIntersection>::NonconformingAccessorPatch2D;
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_intersection = NQuadIntersection;
};

} // namespace specfem::test_fixture::impl
