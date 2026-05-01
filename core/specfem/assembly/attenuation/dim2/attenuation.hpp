#pragma once

#include "specfem/assembly/attenuation.hpp"
#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/info.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/constants.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/mesh/attenuation_config.hpp"
#include "specfem/mesh/dim2/materials/materials.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

template <>
struct Attenuation<specfem::element::dimension_tag::dim2>
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
  constexpr static int N_SLS =
      specfem::constants::N_SLS; ///< Number of standard linear solids

  ///@}

  // Number of GLL points and elements
  int ngllz; ///< Number of GLL points in the z-direction
  int ngllx; ///< Number of GLL points in the x-direction
  int nspec; ///< Total number of spectral elements

  // Attenuation parameters
  specfem::units::Hertz f0; ///< Reference frequency
  type_real deltat;         ///< Time step size

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

  // Device-accessible Runge-Kutta factors for use in device kernels
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
               Kokkos::DefaultExecutionSpace>
      d_alpha_rk;
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
               Kokkos::DefaultExecutionSpace>
      d_beta_rk;
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
               Kokkos::DefaultExecutionSpace>
      d_gamma_rk;

  static constexpr auto attenuation_medium_combinations =
      specfem::tag_dispatch::dimension_set<dimension_tag>{} *
      MEDIUM_SET(elastic_psv) * PROPERTY_SET(isotropic) *
      ATTENUATION_SET(constant_isotropic);

  template <typename TagsType>
  using AttenuationMediumTemplateType =
      specfem::assembly::impl::attenuation_medium<
          TagsType::dimension_tag, TagsType::medium_tag, TagsType::property_tag,
          TagsType::attenuation_tag>;

  specfem::tag_dispatch::TypedStorage<AttenuationMediumTemplateType,
                                      decltype(attenuation_medium_combinations)>
      attenuation_storage; ///< Storage for attenuation medium containers keyed
                           ///< by (medium, property, attenuation) combination

  Attenuation() = default;

  Attenuation(
      const specfem::mesh::attenuation_config &config, const type_real deltat,
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
      const specfem::units::Hertz fc, const specfem::units::Hertz f0,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
      const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
          &tau_sigma);

  /**
   * @brief Access the attenuation_medium for a given medium and property.
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  KOKKOS_INLINE_FUNCTION const specfem::assembly::impl::attenuation_medium<
      dimension_tag, MediumTag, PropertyTag,
      specfem::element::attenuation_tag::constant_isotropic> &
  get_container() const {
    using Key = specfem::tags::Tags<
        dimension_tag, MediumTag, PropertyTag,
        specfem::element::attenuation_tag::constant_isotropic>;
    return attenuation_storage.template get<Key>();
  }

  /**
   * @brief Copy attenuation data to host mirrors.
   */
  void copy_to_host() {
    specfem::tag_dispatch::for_each(
        attenuation_medium_combinations, [&]<typename TagsType>() {
          attenuation_storage.template get<TagsType>().copy_to_host();
        });
  }

  /**
   * @brief Copy attenuation data to device views.
   */
  void copy_to_device() {
    specfem::tag_dispatch::for_each(
        attenuation_medium_combinations, [&]<typename TagsType>() {
          attenuation_storage.template get<TagsType>().copy_to_device();
        });
  }
};

} // namespace specfem::assembly
