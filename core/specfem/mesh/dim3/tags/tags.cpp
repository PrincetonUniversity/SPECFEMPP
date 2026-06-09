#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"

specfem::mesh::tags<specfem::element::dimension_tag::dim3>::tags(
    const int nspec, specfem::mesh::materials<dimension_tag> &materials,
    const specfem::mesh::boundaries<dimension_tag> &boundaries) {

  this->nspec = nspec;

  this->tags_container =
      Kokkos::View<specfem::mesh::impl::tags_container *, Kokkos::HostSpace>(
          "specfem::mesh::tags::tags", this->nspec);

  std::vector<specfem::element::boundary_tag_container> bc(nspec);

  // All faces in the absorbing_boundary sub-struct → stacey
  const auto &abs = boundaries.absorbing_boundary;
  for (int i = 0; i < abs.nelements; ++i) {
    bc[abs.index_mapping(i)] += specfem::element::boundary_tag::stacey;
  }

  // Top-surface faces in acoustic_free_surface sub-struct → acoustic BC only
  // for acoustic elements (elastic elements at the top get natural Neumann BC)
  const auto &fs = boundaries.acoustic_free_surface;
  for (int i = 0; i < fs.nelem_acoustic_surface; ++i) {
    const int ispec = fs.index_mapping(i);
    const auto [medium_tag, property_tag, attenuation_tag] =
        materials.get_material_type(ispec);
    if (medium_tag == specfem::element::medium_tag::acoustic) {
      bc[ispec] += specfem::element::boundary_tag::acoustic_free_surface;
    }
  }

  for (int ispec = 0; ispec < nspec; ++ispec) {
    const auto [medium_tag, property_tag, attenuation_tag] =
        materials.get_material_type(ispec);
    this->tags_container(ispec) = { medium_tag, property_tag, attenuation_tag,
                                    bc[ispec].get_tag() };
  }
}
