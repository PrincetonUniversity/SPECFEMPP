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
    template <specfem::element::medium_tag, specfem::element::property_tag>
    class containers_type>
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

  containers_type<specfem::element::medium_tag::elastic_sv,
                  specfem::element::property_tag::isotropic>
      elastic_isotropic; ///< Elastic isotropic material values

  containers_type<specfem::element::medium_tag::elastic_sv,
                  specfem::element::property_tag::anisotropic>
      elastic_anisotropic; ///< Elastic isotropic material values

  containers_type<specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic>
      acoustic_isotropic; ///< Acoustic isotropic material values

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
    if constexpr ((MediumTag == specfem::element::medium_tag::elastic_sv) &&
                  (PropertyTag == specfem::element::property_tag::isotropic)) {
      return elastic_isotropic;
    } else if constexpr ((MediumTag ==
                          specfem::element::medium_tag::elastic_sv) &&
                         (PropertyTag ==
                          specfem::element::property_tag::anisotropic)) {
      return elastic_anisotropic;
    } else if constexpr ((MediumTag ==
                          specfem::element::medium_tag::acoustic) &&
                         (PropertyTag ==
                          specfem::element::property_tag::isotropic)) {
      return acoustic_isotropic;
    } else {
      static_assert("Material type not implemented");
    }
  }

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    elastic_isotropic.copy_to_host();
    elastic_anisotropic.copy_to_host();
    acoustic_isotropic.copy_to_host();
  }

  void copy_to_device() {
    elastic_isotropic.copy_to_device();
    elastic_anisotropic.copy_to_device();
    acoustic_isotropic.copy_to_device();
  }
};

} // namespace impl
} // namespace compute
} // namespace specfem
