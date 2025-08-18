#include "mesh/dim2/tags/tags.hpp"
#include "enumerations/dimension.hpp"
#include "kokkos_abstractions.h"

specfem::mesh::tags<specfem::dimension::type::dim2>::tags(
    const specfem::mesh::materials<specfem::dimension::type::dim2> &materials,
    const specfem::mesh::boundaries<specfem::dimension::type::dim2>
        &boundaries) {

  this->nspec = materials.material_index_mapping.extent(0);

  this->tags_container =
      specfem::kokkos::HostView1d<specfem::mesh::impl::tags_container>(
          "specfem::mesh::tags::tags", this->nspec);

  std::vector<specfem::element::boundary_tag_container> boundary_tag(
      this->nspec);

  const auto &absorbing_boundary = boundaries.absorbing_boundary;
  for (int i = 0; i < absorbing_boundary.nelements; ++i) {
    const int ispec = absorbing_boundary.index_mapping(i);
    boundary_tag[ispec] += specfem::element::boundary_tag::stacey;
  }

  const auto &acoustic_free_surface = boundaries.acoustic_free_surface;
  for (int i = 0; i < acoustic_free_surface.nelem_acoustic_surface; ++i) {
    const int ispec = acoustic_free_surface.index_mapping(i);
    const auto &material_specification =
        materials.material_index_mapping(ispec);
    if (material_specification.type != specfem::element::medium_tag::acoustic) {
      throw std::invalid_argument(
          "Error: Acoustic free surface boundary is not an acoustic element");
    }
    boundary_tag[ispec] +=
        specfem::element::boundary_tag::acoustic_free_surface;
  }

  for (int ispec = 0; ispec < nspec; ispec++) {
    const auto &material_specification =
        materials.material_index_mapping(ispec);
    const auto medium_tag = material_specification.type;
    const auto property_tag = material_specification.property;

    this->tags_container(ispec).medium_tag = medium_tag;
    this->tags_container(ispec).property_tag = property_tag;
    this->tags_container(ispec).boundary_tag = boundary_tag[ispec].get_tag();
  }
}
