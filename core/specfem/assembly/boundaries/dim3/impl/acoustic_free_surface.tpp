#pragma once

#include <map>
#include <stdexcept>
#include <vector>

#include "acoustic_free_surface.hpp"
#include "specfem/macros.hpp"
#include "utilities.hpp"

template <specfem::simulation::model ModelTag>
specfem::assembly::boundaries_impl::acoustic_free_surface<
    specfem::element::dimension_tag::dim3>::
    acoustic_free_surface(
        const int nspec, const int ngllz, const int nglly, const int ngllx,
        const specfem::mesh::mesh<ModelTag> &mesh,
        const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
        const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
        std::vector<specfem::element::boundary_tag_container>
            &element_boundary_tags) {

  // Identify acoustic free surface elements: acoustic elements in the
  // mesh.boundaries.acoustic_free_surface sub-struct (Z_MAX faces).

  const auto &fs = mesh.boundaries.acoustic_free_surface;

  // Build a map from compute element index to sub-struct face indices
  std::map<int, std::vector<int>> ispec_to_top_faces;

  for (int i = 0; i < fs.nelem_acoustic_surface; ++i) {
    const int ispec_mesh = fs.index_mapping(i);
    const int ispec_compute = mesh_assembly.h_mesh_to_compute(ispec_mesh);

    const auto medium_tag = mesh.tags.tags_container(ispec_mesh).medium_tag;
    if (medium_tag != specfem::element::medium_tag::acoustic)
      continue;

    ispec_to_top_faces[ispec_compute].push_back(i);
  }

  const int total_acfree_elements = ispec_to_top_faces.size();

  // Initialize index mappings to -1 (no acoustic free surface)
  for (int ispec = 0; ispec < nspec; ++ispec) {
    boundary_index_mapping(ispec) = -1;
  }

  // Assign contiguous local indices to free surface elements
  int total_indices = 0;
  for (auto &kv : ispec_to_top_faces) {
    boundary_index_mapping(kv.first) = total_indices;
    ++total_indices;
  }

  if (total_indices != total_acfree_elements) {
    KOKKOS_ABORT_WITH_LOCATION(
        "Error: Mismatch in total acoustic free surface elements");
  }

  // Verify contiguous mapping (required for SIMD correctness)
  for (int ispec = 1; ispec < nspec; ++ispec) {
    if ((boundary_index_mapping(ispec) == -1) ||
        (boundary_index_mapping(ispec - 1) == -1))
      continue;
    if (boundary_index_mapping(ispec) != boundary_index_mapping(ispec - 1) + 1)
      throw std::runtime_error("Boundary index mapping is not contiguous");
  }

  // Allocate views
  this->quadrature_point_boundary_tag =
      BoundaryTagView("specfem::assembly::impl::boundaries::"
                      "acoustic_free_surface::quadrature_point_boundary_tag",
                      total_indices, ngllz, nglly, ngllx);
  this->h_quadrature_point_boundary_tag =
      Kokkos::create_mirror_view(quadrature_point_boundary_tag);

  // Populate boundary tags per quadrature point
  for (auto &kv : ispec_to_top_faces) {
    const int ispec_compute = kv.first;
    const int local_index = boundary_index_mapping(ispec_compute);

    element_boundary_tags[ispec_compute] +=
        specfem::element::boundary_tag::acoustic_free_surface;

    for (int i : kv.second) {
      const auto face_type = fs.type(i);
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix) {
            if (is_on_boundary(face_type, iz, iy, ix, ngllz, nglly, ngllx)) {
              this->h_quadrature_point_boundary_tag(local_index, iz, iy, ix) +=
                  specfem::element::boundary_tag::acoustic_free_surface;
            }
          }
        }
      }
    }
  }

  Kokkos::deep_copy(quadrature_point_boundary_tag,
                    h_quadrature_point_boundary_tag);
}
