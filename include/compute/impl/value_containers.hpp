#pragma once

#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {
/**
 * @brief Values for every quadrature point in the finite element mesh
 *
 */
template <template <specfem::element::medium_tag,
                    specfem::element::property_tag> class containers_type>
struct value_containers {

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  int nspec; ///< Total number of spectral elements
  int ngllz; ///< Number of quadrature points in z dimension
  int ngllx; ///< Number of quadrature points in x dimension

  IndexViewType property_index_mapping; ///< View to store property index
                                        ///< mapping
  IndexViewType::HostMirror h_property_index_mapping; ///< Host mirror of
                                                      ///< property index
                                                      ///< mapping

#define GENERATE_CONTAINER_NAME(POSTFIX, DIMENSION_TAG, MEDIUM_TAG,            \
                                PROPERTY_TAG)                                  \
  containers_type<MEDIUM_TAG, PROPERTY_TAG> value_##POSTFIX;

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(WHERE(DIMENSION_TAG_DIM2)
                                          WHERE(MEDIUM_TAG_ELASTIC_SV,
                                                MEDIUM_TAG_ELASTIC_SH,
                                                MEDIUM_TAG_ACOUSTIC)
                                              WHERE(PROPERTY_TAG_ISOTROPIC,
                                                    PROPERTY_TAG_ANISOTROPIC),
                                      GENERATE_CONTAINER_NAME);

#undef GENERATE_CONTAINER_NAME

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
      constexpr containers_type<MediumTag, PropertyTag> const &
      get_container() const {

    CALL_CODE_FOR_ALL_MATERIAL_SYSTEMS(
        CAPTURE(value) WHERE(DIMENSION_TAG_DIM2) WHERE(
            MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
        if constexpr (_medium_tag_ == MediumTag &&
                      _property_tag_ == PropertyTag) { return _value_; });

    Kokkos::abort("Invalid material type detected in value containers");

    /// code path should never be reached

    auto return_value = new containers_type<MediumTag, PropertyTag>();

    return *return_value;
  }

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    CALL_CODE_FOR_ALL_MATERIAL_SYSTEMS(
        CAPTURE(value) WHERE(DIMENSION_TAG_DIM2) WHERE(
            MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
        _value_.copy_to_host(););
  }

  void copy_to_device() {
    CALL_CODE_FOR_ALL_MATERIAL_SYSTEMS(
        CAPTURE(value) WHERE(DIMENSION_TAG_DIM2) WHERE(
            MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
        _value_.copy_to_device(););
  }
};

} // namespace impl
} // namespace compute
} // namespace specfem
