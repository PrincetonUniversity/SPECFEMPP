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
 * @brief 2D per-GLL-point field derivative storage for attenuation strain
 * bookkeeping.
 *
 * Holds one field_derivative_medium per medium tag. But selects whether to
 * store field_derivative values based on the, e.g., attenuation tag: for
 * attenuation_none, the
 *
 * Follows the same pattern as specfem::assembly::properties<dim2>.
 */
template <>
struct FieldDerivativeStorage<specfem::element::dimension_tag::dim2>
    : specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::field_derivatives,
          specfem::element::dimension_tag::dim2> {

  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::field_derivatives,
      specfem::element::dimension_tag::dim2>;

  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;

  static constexpr auto field_derivative_medium_combinations =
      specfem::tag_dispatch::dimension_set<dimension_tag>{} *
      MEDIUM_SET(elastic_psv);

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

  /**
   * @brief Construct and allocate storage for all non-none element
   * combinations.
   *
   * @param element_types Element type information (provides per-combination
   *                      element index lists).
   * @param nspec_global  Total spectral elements (for ispec_to_compact
   * mapping).
   * @param ngllz         GLL points in z-direction.
   * @param ngllx         GLL points in x-direction.
   */
  FieldDerivativeStorage(
      const specfem::assembly::element_types<
          specfem::element::dimension_tag::dim2> &element_types,
      const int nspec_global, const int ngllz, const int ngllx);

  /**
   * @brief Access the field_derivative_medium for a given tag combination.
   */
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
