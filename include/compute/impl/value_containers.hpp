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
template <
    template <specfem::element::medium_tag, specfem::element::property_tag,
              typename Enable = void> class containers_type>
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

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      DECLARE(((containers_type, (_MEDIUM_TAG_, _PROPERTY_TAG_)), value)))

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

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
        CAPTURE(value) {
          if constexpr (_medium_tag_ == MediumTag &&
                        _property_tag_ == PropertyTag) {
            return _value_;
          }
        })

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
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
        CAPTURE(value) { _value_.copy_to_host(); })
  }

  void copy_to_device() {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
        CAPTURE(value) { _value_.copy_to_device(); })
  }
};

} // namespace impl
} // namespace compute
} // namespace specfem
