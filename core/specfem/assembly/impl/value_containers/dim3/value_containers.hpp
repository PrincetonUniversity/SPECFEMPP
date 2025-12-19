#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::impl {

template <template <specfem::dimension::type, specfem::element::medium_tag,
                    specfem::element::property_tag> class containers_type>
struct value_containers<specfem::dimension::type::dim3, containers_type> {

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  int nspec; ///< Total number of spectral elements
  int ngllz; ///< Number of quadrature points in z dimension
  int ngllx; ///< Number of quadrature points in x dimension
  int nglly; ///< Number of quadrature points in y dimension

  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  IndexViewType property_index_mapping; ///< View to store property index
                                        ///< mapping
  IndexViewType::HostMirror h_property_index_mapping; ///< Host mirror of
                                                      ///< property index
                                                      ///< mapping

  template <bool on_device>
  KOKKOS_INLINE_FUNCTION constexpr
      typename std::conditional<on_device, IndexViewType,
                                IndexViewType::HostMirror>::type
      get_property_index_mapping() const {
    if constexpr (on_device) {
      return property_index_mapping;
    } else {
      return h_property_index_mapping;
    }
  }

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC),
                       PROPERTY_TAG(ISOTROPIC)),
                      DECLARE(((containers_type, (_DIMENSION_TAG_, _MEDIUM_TAG_,
                                                  _PROPERTY_TAG_)),
                               value)))

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  value_containers() = default;

  /**
   * @brief Returns the material_kernel for a given medium and property
   *
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  KOKKOS_INLINE_FUNCTION
      constexpr containers_type<dimension_tag, MediumTag, PropertyTag> const &
      get_container() const {

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC)),
        CAPTURE(value) {
          if constexpr (_medium_tag_ == MediumTag &&
                        _property_tag_ == PropertyTag) {
            return _value_;
          }
        })

    Kokkos::abort("Invalid material type detected in value containers");

    /// code path should never be reached

    auto return_value =
        new containers_type<dimension_tag, MediumTag, PropertyTag>();

    return *return_value;
  }

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC)),
        CAPTURE(value) { _value_.copy_to_host(); })
  }

  void copy_to_device() {
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC)),
        CAPTURE(value) { _value_.copy_to_device(); })
  }
};

} // namespace specfem::assembly::impl
