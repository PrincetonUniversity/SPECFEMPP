#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/field_derivative_storage.hpp"
#include "specfem/assembly/field_derivative_storage/impl/field_derivative_medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @brief 2D per-GLL-point field derivative storage for attenuation strain
 * bookkeeping.
 *
 * Holds one field_derivative_medium per (medium, property, attenuation) tag
 * combination. For attenuation_none combinations the sub-struct is empty
 * (zero overhead). For constant_isotropic combinations compact storage is
 * allocated for the matching elements.
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

  // One field_derivative_medium per (medium, property, attenuation)
  // combination. Attenuation_none → empty struct. Constant_isotropic → compact
  // storage.
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV),
                       PROPERTY_TAG(ISOTROPIC),
                       ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
                      DECLARE(((specfem::assembly::field_derivative_storage::
                                    impl::field_derivative_medium,
                                (_DIMENSION_TAG_, _MEDIUM_TAG_, _PROPERTY_TAG_,
                                 _ATTENUATION_TAG_)),
                               fd_medium)))

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
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag,
            specfem::element::attenuation_tag AttenuationTag>
  KOKKOS_INLINE_FUNCTION constexpr specfem::assembly::field_derivative_storage::
      impl::field_derivative_medium<specfem::element::dimension_tag::dim2,
                                    MediumTag, PropertyTag,
                                    AttenuationTag> const &
      get_medium() const {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV),
                         PROPERTY_TAG(ISOTROPIC),
                         ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
                        CAPTURE(fd_medium) {
                          if constexpr (_medium_tag_ == MediumTag &&
                                        _property_tag_ == PropertyTag &&
                                        _attenuation_tag_ == AttenuationTag) {
                            return _fd_medium_;
                          }
                        })
    Kokkos::abort(
        "Invalid tag combination in FieldDerivativeStorage::get_medium");
    SUPPRESS_UNREACHABLE(return {};)
  }

  /**
   * @brief Non-const overload for store_on_device access.
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag,
            specfem::element::attenuation_tag AttenuationTag>
  KOKKOS_INLINE_FUNCTION constexpr specfem::assembly::field_derivative_storage::
      impl::field_derivative_medium<specfem::element::dimension_tag::dim2,
                                    MediumTag, PropertyTag, AttenuationTag> &
      get_medium() {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV),
                         PROPERTY_TAG(ISOTROPIC),
                         ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
                        CAPTURE(fd_medium) {
                          if constexpr (_medium_tag_ == MediumTag &&
                                        _property_tag_ == PropertyTag &&
                                        _attenuation_tag_ == AttenuationTag) {
                            return _fd_medium_;
                          }
                        })
    Kokkos::abort(
        "Invalid tag combination in FieldDerivativeStorage::get_medium");
    SUPPRESS_UNREACHABLE(return {};)
  }

  void copy_to_host() {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV),
                         PROPERTY_TAG(ISOTROPIC),
                         ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
                        CAPTURE(fd_medium) { _fd_medium_.copy_to_host(); })
  }

  void copy_to_device() {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV),
                         PROPERTY_TAG(ISOTROPIC),
                         ATTENUATION_TAG(CONSTANT_ISOTROPIC)),

                        CAPTURE(fd_medium) { _fd_medium_.copy_to_device(); })
  }
};

} // namespace specfem::assembly
