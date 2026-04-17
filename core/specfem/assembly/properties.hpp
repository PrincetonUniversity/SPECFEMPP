#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/impl/domain_properties.tpp"
#include "specfem/assembly/impl/value_containers.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace specfem::assembly {

/**
 * @brief Spectral element material properties data container
 *
 * This class provides storage and data access functions for the material
 * properties associated with spectral elements in spectral element mesh.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 */
template <specfem::element::dimension_tag DimensionTag>
struct properties
    : public impl::value_containers<DimensionTag, impl::domain_properties> {

  properties() = default;

  properties(
      const specfem::assembly::element_types<DimensionTag> &element_types,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::mesh::materials<DimensionTag> &materials,
      bool has_gll_model)
      : impl::value_containers<DimensionTag, impl::domain_properties>(
            element_types.nspec, element_types.element_grid,
            "specfem::assembly::properties::property_index_mapping",
            [&element_types, &mesh, &materials, has_gll_model](auto h_prop) {
              return [&element_types, &mesh, &materials, has_gll_model,
                      h_prop]<typename TagsType>() {
                return specfem::assembly::impl::domain_properties<
                    TagsType::dimension_tag, TagsType::medium_tag,
                    TagsType::property_tag>(
                    element_types.get_elements_on_host(
                        TagsType::medium_tag, TagsType::property_tag,
                        TagsType::attenuation_tag),
                    mesh, materials, has_gll_model, h_prop);
              };
            }) {}
};

template <typename IndexViewType, typename PointPropertiesType>
void max(
    const IndexViewType &ispecs,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim2>
        &properties,
    PointPropertiesType &point_properties) {

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;

  constexpr bool on_device =
      std::is_same<typename IndexViewType::execution_space,
                   Kokkos::DefaultExecutionSpace>::value;

  static_assert(PointPropertiesType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
                "Only 2D properties are supported");

  IndexViewType local_ispecs("local_ispecs", ispecs.extent(0));

  const auto index_mapping = properties.get_property_index_mapping<on_device>();

  Kokkos::parallel_for(
      "local_work_items",
      Kokkos::RangePolicy<typename IndexViewType::execution_space>(
          0, ispecs.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        local_ispecs(i) =
            index_mapping(ispecs(i)); // Map the ispec to the property index
      });

  Kokkos::fence(); // Ensure the above parallel for is complete before
                   // proceeding

  properties.get_container<MediumTag, PropertyTag>().max(local_ispecs,
                                                         point_properties);

  return;
}

} // namespace specfem::assembly

extern template struct specfem::assembly::properties<
    specfem::element::dimension_tag::dim2>;
extern template struct specfem::assembly::properties<
    specfem::element::dimension_tag::dim3>;

// Data access functions
#include "properties/load_on_device.hpp"
#include "properties/load_on_host.hpp"
#include "properties/store_on_host.hpp"
