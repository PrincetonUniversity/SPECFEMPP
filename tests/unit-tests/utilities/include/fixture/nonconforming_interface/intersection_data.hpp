#pragma once

#include "enumerations/coupled_interface.hpp"
#include "enumerations/dimension.hpp"
#include "initializers.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem_setup.hpp"

#include <type_traits>
static constexpr specfem::dimension::type dimension_tag_ =
    specfem::dimension::type::dim2;
namespace specfem::test_fixture {

template <specfem::interface::interface_tag InterfaceTag, typename... Accessors>
struct IntersectionDataPack
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::nonconforming_interface,
          dimension_tag_, false>,
      public Accessors... {
  static constexpr specfem::connections::type connection_tag =
      specfem::connections::type::nonconforming;
  static constexpr specfem::interface::interface_tag interface_tag =
      InterfaceTag;
  static constexpr specfem::dimension::type dimension_tag = dimension_tag_;

  constexpr static size_t n_accessors = sizeof...(Accessors);
  using packed_accessors = std::tuple<Accessors...>;

  KOKKOS_INLINE_FUNCTION
  IntersectionDataPack(const Accessors &...accessors)
      : Accessors(accessors)... {};

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION type_real operator()(Indices... indices) const =
      delete;

  static std::string description() {
    return std::string("IntersectionAccessorPack:\n") +
           ((specfem::test_fixture::impl::description<Accessors>::get(2) +
             "\n") +
            ...);
  }
  static std::string name() {
    return std::string("IntersectionAccessorPack (...)");
  }
};

template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename Initializer,
          specfem::data_access::DataClassType... PackedTypes>
struct IntersectionDataPack2D
    : specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::dimension::type::dim2, false /*UseSIMD*/>,
      specfem::test_fixture::NonconformingAccessorPatch2D<
          InterfaceTag, BoundaryTag, Initializer, PackedTypes>... {
  constexpr static auto connection_tag =
      specfem::connections::type::nonconforming;
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;

  template <specfem::data_access::DataClassType DataClass>
  specfem::test_fixture::NonconformingAccessorPatch2D<InterfaceTag, BoundaryTag,
                                                      Initializer, DataClass> &
  get_component() {
    return static_cast<specfem::test_fixture::NonconformingAccessorPatch2D<
        InterfaceTag, BoundaryTag, Initializer, DataClass> >(*this);
  }

  KOKKOS_INLINE_FUNCTION IntersectionDataPack2D() = default;
  IntersectionDataPack2D(const std::string &name)
      : specfem::test_fixture::NonconformingAccessorPatch2D<
            InterfaceTag, BoundaryTag, Initializer, PackedTypes>(name)... {}

  static std::string description() {
    return std::string("IntersectionDataPack2D:\n") +
           specfem::test_fixture::impl::description<Initializer>::get(2) +
           ((std::string("\n  - ") + std::to_string(PackedTypes)) + ...);
  }
  static std::string name() {
    return specfem::test_fixture::impl::name<Initializer>::get();
  }
};

namespace IntersectionDataInitializer2D {

template <specfem::interface::interface_tag InterfaceTag,
          typename TransferFunctionInitializer,
          typename IntersectionFunctionInitializer>
struct CoupledTransferAndNormal : IntersectionDataInitializer2D {
  static constexpr specfem::interface::interface_tag interface_tag =
      InterfaceTag;
  using TransferFunction = TransferFunctionInitializer;
  static_assert(
      std::is_base_of_v<
          TransferFunctionInitializer2D::TransferFunctionInitializer2D,
          TransferFunctionInitializer>,
      "CoupledTransferAndNormal needs an TransferFunctionInitializer2D!");
  using IntersectionFunction = IntersectionFunctionInitializer;
  static_assert(
      std::is_base_of_v<
          IntersectionFunctionInitializer2D::IntersectionFunctionInitializer2D,
          IntersectionFunctionInitializer>,
      "CoupledTransferAndNormal needs an IntersectionFunctionInitializer2D!");
  static_assert(
      IntersectionFunction::num_edges == TransferFunction::num_edges,
      "TransferFunction and IntersectionFunction have incompatible num_edges!");
  static constexpr int num_edges = IntersectionFunction::num_edges;
  static_assert(IntersectionFunction::nquad_intersection ==
                    TransferFunction::nquad_intersection,
                "TransferFunction and IntersectionFunction have "
                "incompatible quadrature points!");

  static constexpr int nquad_intersection =
      IntersectionFunction::nquad_intersection;
  static constexpr int nquad_edge = TransferFunction::nquad_edge;

private:
  using TransferArrayType = std::array<
      std::array<std::array<type_real, nquad_intersection>, nquad_edge>,
      num_edges>;
  using IntersectionNormalArrayType =
      std::array<std::array<std::array<type_real, 2>, nquad_intersection>,
                 num_edges>;

public:
  static TransferArrayType init_transfer_function_coupled() {
    return TransferFunction::init_transfer_function();
  }
  static IntersectionNormalArrayType init_intersection_normal() {
    return TransferFunction::init_transfer_function();
  }
};

} // namespace IntersectionDataInitializer2D

} // namespace specfem::test_fixture
