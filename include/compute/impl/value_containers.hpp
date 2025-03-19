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

#define GENERATE_CONTAINER_NAME(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,       \
                                POSTFIX)                                       \
  containers_type<MEDIUM_TAG, PROPERTY_TAG> value_##POSTFIX;

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS2(GENERATE_CONTAINER_NAME,
                                       WHERE(DIMENSION_TAG_DIM2)
                                           WHERE(MEDIUM_TAG_ELASTIC_SV,
                                                 MEDIUM_TAG_ELASTIC_SH,
                                                 MEDIUM_TAG_ACOUSTIC)
                                               WHERE(PROPERTY_TAG_ISOTROPIC,
                                                     PROPERTY_TAG_ANISOTROPIC));

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

#define GET_CONTAINER(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)                 \
  if constexpr ((GET_TAG(MEDIUM_TAG) == MediumTag) &&                          \
                (GET_TAG(PROPERTY_TAG) == PropertyTag)) {                      \
    return CREATE_VARIABLE_NAME(value, GET_NAME(DIMENSION_TAG),                \
                                GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)); \
  }

    CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
        GET_CONTAINER,
        WHERE(DIMENSION_TAG_DIM2) WHERE(
            MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC));
#undef GET_CONTAINER

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
#define COPY_TO_HOST(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)                  \
  CREATE_VARIABLE_NAME(value, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),   \
                       GET_NAME(PROPERTY_TAG))                                 \
      .copy_to_host();

    CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
        COPY_TO_HOST,
        WHERE(DIMENSION_TAG_DIM2) WHERE(
            MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC));

#undef COPY_TO_HOST
  }

  void copy_to_device() {
#define COPY_TO_DEVICE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)                \
  CREATE_VARIABLE_NAME(value, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),   \
                       GET_NAME(PROPERTY_TAG))                                 \
      .copy_to_device();

    CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
        COPY_TO_DEVICE,
        WHERE(DIMENSION_TAG_DIM2) WHERE(
            MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC));

#undef COPY_TO_DEVICE
  }
};

} // namespace impl
} // namespace compute
} // namespace specfem
