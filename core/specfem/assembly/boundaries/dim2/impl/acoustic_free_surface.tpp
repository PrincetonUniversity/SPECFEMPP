#pragma once

#include <map>
#include <stdexcept>
#include <vector>

#include "acoustic_free_surface.hpp"
#include "macros.hpp"
#include "utilities.hpp"

specfem::assembly::boundaries_impl::acoustic_free_surface<specfem::dimension::type::dim2>::
    acoustic_free_surface(
        const int nspec, const int ngllz, const int ngllx,
        const specfem::mesh::acoustic_free_surface<dimension_tag>
            &acoustic_free_surface,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
        std::vector<specfem::element::boundary_tag_container>
            &element_boundary_tags) {

  // We need to make sure that boundary index mapping maps every spectral
  // element index to the corresponding index within
  // quadrature_point_boundary_tag

  // mesh.acoustic_free_surface.ispec_acoustic_surface stores the ispec for
  // every acoustic free surface. At the corners of the mesh, multiple surfaces
  // belong to the same ispec. The first part of the code assigns a unique index
  // to each ispec.

  // For SIMD loads we need to ensure that there is a contiguous mapping within
  // ispec and index of boundary_index_mapping i.e. boundary_index_mapping(ispec
  // +1) - boundary_index_mapping(ispec) = 1

  // -------------------------------------------------------------------

  // Create a map from ispec to index in acoustic_free_surface

  const int nelements = acoustic_free_surface.nelem_acoustic_surface;
  std::map<int, std::vector<int> > ispec_to_acoustic_surface;

  for (int i = 0; i < nelements; ++i) {
    const int ispec_mesh = acoustic_free_surface.index_mapping(i);
    const int ispec_compute = mesh.mesh_to_compute(ispec_mesh);
    if (ispec_to_acoustic_surface.find(ispec_compute) ==
        ispec_to_acoustic_surface.end()) {
      ispec_to_acoustic_surface[ispec_compute] = { i };
    } else {
      ispec_to_acoustic_surface[ispec_compute].push_back(i);
    }
  }

  const int total_acfree_surface_elements = ispec_to_acoustic_surface.size();

  // -------------------------------------------------------------------

  // Initialize all index mappings to -1
  for (int ispec = 0; ispec < nspec; ++ispec) {
    boundary_index_mapping(ispec) = -1;
  }

  // -------------------------------------------------------------------

  // Assign boundary index mapping
  int total_indices = 0;
  for (auto &map : ispec_to_acoustic_surface) {
    const int ispec = map.first;
    boundary_index_mapping(ispec) = total_indices;
    ++total_indices;
  }

  ASSERT(total_indices == total_acfree_surface_elements,
         "Total indices do not match");

  // -------------------------------------------------------------------

  // Make sure the index mapping is contiguous
  for (int ispec = 0; ispec < nspec; ++ispec) {
    if (ispec == 0)
      continue;

    if ((boundary_index_mapping(ispec) == -1) ||
        (boundary_index_mapping(ispec - 1) == -1))
      continue;

    if (boundary_index_mapping(ispec) !=
        boundary_index_mapping(ispec - 1) + 1) {
      throw std::runtime_error("Boundary index mapping is not contiguous");
    }
  }

  // -------------------------------------------------------------------

  // Initialize boundary tags
  this->quadrature_point_boundary_tag =
      BoundaryTagView("specfem::assembly::impl::boundaries::"
                      "acoustic_free_surface::quadrature_point_boundary_tag",
                      total_indices, ngllz, ngllx);

  this->h_quadrature_point_boundary_tag =
      Kokkos::create_mirror_view(quadrature_point_boundary_tag);

  // Assign boundary tags

  for (auto &map : ispec_to_acoustic_surface) {
    const int ispec = map.first;
    const auto &indices = map.second;
    for (auto &index : indices) {
      const auto type = acoustic_free_surface.type(index);
      const int index_compute = boundary_index_mapping(ispec);
      element_boundary_tags[ispec] +=
          specfem::element::boundary_tag::acoustic_free_surface;

      // Assign boundary tag to each quadrature point
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          if (is_on_boundary(type, iz, ix, ngllz, ngllx)) {
            this->h_quadrature_point_boundary_tag(index_compute, iz, ix) +=
                specfem::element::boundary_tag::acoustic_free_surface;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(quadrature_point_boundary_tag,
                    h_quadrature_point_boundary_tag);
}
