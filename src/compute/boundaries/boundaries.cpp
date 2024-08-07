
#include "compute/boundaries/boundaries.hpp"

specfem::compute::boundaries::boundaries(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::mesh::mesh &mesh,
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::compute::quadrature &quadrature,
    const specfem::compute::properties &properties,
    const specfem::compute::partial_derivatives &partial_derivatives)
    : boundary_tags("specfem::compute::boundaries::boundary_tags", nspec, ngllz,
                    ngllx),
      h_boundary_tags(Kokkos::create_mirror_view(boundary_tags)),
      acoustic_free_surface_index_mapping(
          "specfem::compute::boundaries::acoustic_free_surface_index_mapping",
          nspec),
      h_acoustic_free_surface_index_mapping(
          Kokkos::create_mirror_view(acoustic_free_surface_index_mapping)),
      stacey_index_mapping("specfem::compute::boundaries::stacey_index_mapping",
                           nspec),
      h_stacey_index_mapping(Kokkos::create_mirror_view(stacey_index_mapping)) {

  this->acoustic_free_surface =
      specfem::compute::impl::boundaries::acoustic_free_surface(
          nspec, ngllz, ngllx, mesh.boundaries.acoustic_free_surface, mapping,
          properties, this->h_acoustic_free_surface_index_mapping,
          this->h_boundary_tags);

  this->stacey = specfem::compute::impl::boundaries::stacey(
      nspec, ngllz, ngllx, mesh.boundaries.absorbing_boundary, mapping,
      quadrature, partial_derivatives, this->h_stacey_index_mapping,
      this->h_boundary_tags);

  // Check if mesh and compute boundary tags match
  for (int ispec = 0; ispec < nspec; ++ispec) {
    const int ispec_compute = mapping.mesh_to_compute(ispec);
    const auto m_boundary_tag = mesh.tags.tags_container(ispec).boundary_tag;
    const auto c_boundary_tag = this->h_boundary_tags(ispec_compute);
    if (m_boundary_tag != c_boundary_tag) {
      throw std::runtime_error("Mesh and compute boundary tags do not match");
    }
  }

  Kokkos::deep_copy(this->boundary_tags, this->h_boundary_tags);

  Kokkos::deep_copy(this->acoustic_free_surface_index_mapping,
                    this->h_acoustic_free_surface_index_mapping);

  Kokkos::deep_copy(this->stacey_index_mapping, this->h_stacey_index_mapping);
}
