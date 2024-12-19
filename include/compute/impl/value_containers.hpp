#pragma once

#include "compute/impl/element_types.hpp"
#include "compute/kernels/impl/material_kernels.hpp"

namespace specfem {
namespace compute {

namespace impl {
/**
 * @brief Values for every quadrature point in the
 * finite element mesh
 *
 */
template <
    template <specfem::element::medium_tag, specfem::element::property_tag>
    class containers_type>
struct value_containers {
  containers_type<specfem::element::medium_tag::elastic,
                  specfem::element::property_tag::isotropic>
      elastic_isotropic; ///< Elastic isotropic material values

  containers_type<specfem::element::medium_tag::elastic,
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
  KOKKOS_INLINE_FUNCTION const containers_type<MediumTag, PropertyTag> &
  get_container() const {
    if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                  (PropertyTag == specfem::element::property_tag::isotropic)) {
      return elastic_isotropic;
    } else if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
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

void compute_number_of_elements_per_medium(
    const int nspec, const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::kokkos::HostView1d<specfem::element::medium_tag>
        &h_medium_tags,
    const specfem::kokkos::HostView1d<specfem::element::property_tag>
        &h_property_tags,
    int &n_elastic_isotropic, int &n_elastic_anisotropic, int &n_acoustic);

} // namespace impl
} // namespace compute
} // namespace specfem
