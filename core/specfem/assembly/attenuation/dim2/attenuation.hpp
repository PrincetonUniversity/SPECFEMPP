#pragma once

#include "specfem/assembly/attenuation.hpp"
#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/info.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/constants.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh/dim2/materials/materials.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

template <>
struct Attenuation<specfem::element::dimension_tag::dim2,
                   specfem::element::attenuation_tag::constant_isotropic>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::attenuation,
          specfem::element::dimension_tag::dim2> {

  /**
   * @name Type Definitions
   *
   */
  ///@{

  /**
   * @brief Base container type providing data access infrastructure
   *
   * @see specfem::data_access::Container
   */
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::attenuation,
      specfem::element::dimension_tag::dim2>;

  /**
   * @brief Kokkos view type for per-element index mapping arrays
   */
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  ///@}

  /**
   * @name Compile Time Definitions
   *
   */
  ///@{

  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim2; ///< Dimension tag for 2D
  constexpr static auto attenuation_tag =
      specfem::element::attenuation_tag::constant_isotropic; ///< Attenuation
                                                             ///< tag for
                                                             ///< constant
                                                             ///< isotropic
  constexpr static int N_SLS =
      specfem::constants::N_SLS; ///< Number of standard linear solids

  ///@}

  // Number of GLL points and elements
  int ngllz; ///< Number of GLL points in the z-direction
  int ngllx; ///< Number of GLL points in the x-direction
  int nspec; ///< Total number of spectral elements

  // Attenuation parameters
  type_real f0;                       ///< Reference frequency
  bool auto_compute_attenuation_band; ///< Whether to auto-compute the
                                      ///< attenuation band

  // Runge-Kutta attenuation factors (one coefficient per SLS mechanism)
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
               Kokkos::DefaultHostExecutionSpace>
      alpha_rk;
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
               Kokkos::DefaultHostExecutionSpace>
      beta_rk;
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
               Kokkos::DefaultHostExecutionSpace>
      gamma_rk;

  // One attenuation_medium member per (medium, property) combination
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV),
                       PROPERTY_TAG(ISOTROPIC),
                       ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
                      DECLARE(((specfem::assembly::impl::attenuation_medium,
                                (_DIMENSION_TAG_, _MEDIUM_TAG_, _PROPERTY_TAG_,
                                 _ATTENUATION_TAG_)),
                               attn_medium)))

  Attenuation() = default;

  Attenuation(
      const type_real reference_frequency, const type_real min_frequency,
      const type_real max_frequency, const bool auto_compute_attenuation_band,
      const type_real deltat,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim2>
          &mesh,
      const specfem::assembly::element_types<
          specfem::element::dimension_tag::dim2> &element_types,
      const specfem::assembly::Info<specfem::element::dimension_tag::dim2>
          &info,
      const specfem::mesh::materials<specfem::element::dimension_tag::dim2>
          &materials);

  void init_memory_variables(
      const specfem::assembly::element_types<
          specfem::element::dimension_tag::dim2> &element_types,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim2>
          &mesh,
      const specfem::mesh::materials<specfem::element::dimension_tag::dim2>
          &materials,
      const type_real fc, const type_real f0,
      const specfem::utilities::FrequencyBand &band,
      const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
          &tau_sigma);

  /**
   * @brief Access the attenuation_medium for a given medium and property.
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  KOKKOS_INLINE_FUNCTION constexpr specfem::assembly::impl::attenuation_medium<
      specfem::element::dimension_tag::dim2, MediumTag, PropertyTag,
      specfem::element::attenuation_tag::constant_isotropic> const &
  get_medium() const {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV),
                         PROPERTY_TAG(ISOTROPIC),
                         ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
                        CAPTURE(attn_medium) {
                          if constexpr (_medium_tag_ == MediumTag &&
                                        _property_tag_ == PropertyTag) {
                            return _attn_medium_;
                          }
                        })
    Kokkos::abort("Invalid medium type detected in attenuation");
    SUPPRESS_TEMPORARY_REF(return {};)
  }

  /**
   * @brief Copy attenuation data to host mirrors.
   */
  void copy_to_host() {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV),
                         PROPERTY_TAG(ISOTROPIC),
                         ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
                        CAPTURE(attn_medium) { _attn_medium_.copy_to_host(); })
  }

  /**
   * @brief Copy attenuation data to device views.
   */
  void copy_to_device() {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV), PROPERTY_TAG(ISOTROPIC),
         ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
        CAPTURE(attn_medium) { _attn_medium_.copy_to_device(); })
  }
};

} // namespace specfem::assembly
