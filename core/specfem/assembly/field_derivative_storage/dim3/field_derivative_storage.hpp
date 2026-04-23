#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/field_derivative_storage.hpp"
#include "specfem/assembly/field_derivative_storage/impl/field_derivative_medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @brief 3D per-GLL-point field derivative storage for attenuation field
 * derivatives bookkeeping.
 *
 * Follows the same pattern as the 2D specialization. For attenuation_none
 * combinations the sub-struct is empty (zero overhead).
 */
template <>
struct FieldDerivativeStorage<specfem::element::dimension_tag::dim3>
    : specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::field_derivatives,
          specfem::element::dimension_tag::dim3> {

  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::field_derivatives,
      specfem::element::dimension_tag::dim3>;

  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  static constexpr auto field_derivative_medium_combinations =
      specfem::tag_dispatch::dimension_set<dimension_tag>{} *
      MEDIUM_SET(elastic);

  template <typename TagsType>
  using FieldDerivativeMediumTemplateType =
      specfem::assembly::field_derivative_storage::impl::
          field_derivative_medium<TagsType::dimension_tag,
                                  TagsType::medium_tag>;

  specfem::tag_dispatch::TypedStorage<FieldDerivativeMediumTemplateType,
                                      decltype(
                                          field_derivative_medium_combinations)>
      field_derivative_storage;

  FieldDerivativeStorage() = default;

  FieldDerivativeStorage(
      const specfem::assembly::element_types<
          specfem::element::dimension_tag::dim3> &element_types,
      const int nspec_global, const int ngllz, const int nglly,
      const int ngllx);

  template <specfem::element::medium_tag MediumTag>
  KOKKOS_INLINE_FUNCTION const specfem::assembly::field_derivative_storage::
      impl::field_derivative_medium<dimension_tag, MediumTag> &
      get_container() const {
    using Key = specfem::tags::Tags<dimension_tag, MediumTag>;
    return field_derivative_storage.template get<Key>();
  }

  void copy_to_host() {
    specfem::tag_dispatch::for_each(
        field_derivative_medium_combinations, [&]<typename TagsType>() {
          field_derivative_storage.template get<TagsType>().copy_to_host();
        });
  }

  void copy_to_device() {
    specfem::tag_dispatch::for_each(
        field_derivative_medium_combinations, [&]<typename TagsType>() {
          field_derivative_storage.template get<TagsType>().copy_to_device();
        });
  }
};

} // namespace specfem::assembly
