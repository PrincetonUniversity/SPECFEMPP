#pragma once

#include "enumerations/coupled_interface.hpp"
#include "enumerations/dimension.hpp"
#include "initializers.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem_setup.hpp"

#include <type_traits>
static constexpr specfem::dimension::type dimension_tag_ =
    specfem::dimension::type::dim2;
namespace specfem::test::fixture {

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
  static constexpr specfem::dimension::type dimension_tag = dimension_tag;

  constexpr static size_t n_accessors = sizeof...(Accessors);
  using packed_accessors = std::tuple<Accessors...>;

  KOKKOS_INLINE_FUNCTION
  IntersectionDataPack(const Accessors &...accessors)
      : Accessors(accessors)... {};

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION type_real operator()(Indices... indices) const =
      delete;

  static std::string description() {
    return std::string("IntersectionDataPack (...)");
  }
};

} // namespace specfem::test::fixture
