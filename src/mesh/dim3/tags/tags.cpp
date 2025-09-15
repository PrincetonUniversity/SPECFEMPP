#include "mesh/dim3/tags/tags.hpp"
#include "enumerations/dimension.hpp"
#include "kokkos_abstractions.h"
#include "mesh/dim3/element_types/element_types.hpp"
#include "mesh/dim3/parameters/parameters.hpp"
#include <Kokkos_Core.hpp>

specfem::mesh::tags<specfem::dimension::type::dim3>::tags(
    const specfem::mesh::element_types<specfem::dimension::type::dim3>
        &element_types,
    const specfem::mesh::boundaries<specfem::dimension::type::dim3> &boundaries,
    const specfem::mesh::parameters<specfem::dimension::type::dim3>
        &parameters) {

  this->nspec = element_types.nspec;

  this->tags_container =
      specfem::kokkos::HostView1d<specfem::mesh::impl::tags_container>(
          "specfem::mesh::tags::tags", this->nspec);

  std::vector<specfem::element::boundary_tag_container> boundary_tag(
      this->nspec);

  if (parameters.stacey_abc) {
    const auto &absorbing_boundary = boundaries.absorbing_boundary;
    for (int i = 0; i < absorbing_boundary.num_abs_boundary_faces; ++i) {
      const int ispec = absorbing_boundary.ispec(i);
      boundary_tag[ispec] += specfem::element::boundary_tag::stacey;
    }
  }

  const auto &acoustic_free_surface = boundaries.acoustic_free_surface;
  for (int i = 0; i < acoustic_free_surface.nelements; ++i) {
    const int ispec = acoustic_free_surface.ispec(i);

    // The acoustic elements at the free surface require special treatment in
    // terms of boundary conditions.
    if (element_types.ispec_type(ispec) ==
        specfem::element::medium_tag::acoustic) {
      // Acoustic free surface requires special treatment
      boundary_tag[ispec] +=
          specfem::element::boundary_tag::acoustic_free_surface;
    } else if (element_types.ispec_type(ispec) ==
               specfem::element::medium_tag::elastic) {
      // No special treatment for elastic free surface
      boundary_tag[ispec] += specfem::element::boundary_tag::none;
    } else {
      std::stringstream message;
      message << "Undefined boundary tagging procedure for ispec " << ispec;
      message << "\n             with medium tag: ";
      message << specfem::element::to_string(element_types.ispec_type(ispec));
      throw std::runtime_error(message.str());
    }
  }

  for (int ispec = 0; ispec < nspec; ispec++) {
    this->tags_container(ispec).medium_tag = element_types.ispec_type(ispec);
    if (parameters.anisotropy) {
      this->tags_container(ispec).property_tag =
          specfem::element::property_tag::anisotropic;
    } else {
      this->tags_container(ispec).property_tag =
          specfem::element::property_tag::isotropic;
    }
    this->tags_container(ispec).boundary_tag = boundary_tag[ispec].get_tag();
  }
}
