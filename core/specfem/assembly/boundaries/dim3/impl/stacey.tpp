#pragma once

#include <array>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "stacey.hpp"
#include "utilities.hpp"
#include <Kokkos_Core.hpp>

specfem::assembly::boundaries_impl::stacey<
    specfem::element::dimension_tag::dim3>::
    stacey(const int nspec, const int ngllz, const int nglly, const int ngllx,
           const specfem::mesh::mesh<dimension_tag> &mesh,
           const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
           const specfem::assembly::jacobian_matrix<dimension_tag>
               &jacobian_matrix,
           const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
           std::vector<specfem::element::boundary_tag_container>
               &element_boundary_tags) {

  // Collect stacey boundary faces: all boundary faces that are NOT acoustic
  // free surface (i.e., not top face on acoustic element).

  const int nfaces = mesh.boundaries.nfaces;

  std::map<int, std::vector<int>> ispec_to_stacey_faces;

  for (int iface = 0; iface < nfaces; ++iface) {
    const auto face_type = mesh.boundaries.face_type(iface);
    const int ispec_mesh = mesh.boundaries.index_mapping(iface);
    const int ispec_compute = mesh_assembly.h_mesh_to_compute(ispec_mesh);

    const auto medium_tag =
        mesh.tags.tags_container(ispec_mesh).medium_tag;

    // Top face on acoustic element is handled as acoustic_free_surface,
    // not stacey.
    if (face_type == specfem::mesh_entity::dim3::type::top &&
        medium_tag == specfem::element::medium_tag::acoustic)
      continue;

    ispec_to_stacey_faces[ispec_compute].push_back(iface);
  }

  const int total_stacey_elements = ispec_to_stacey_faces.size();

  // Initialize index mappings to -1
  for (int ispec = 0; ispec < nspec; ++ispec) {
    boundary_index_mapping(ispec) = -1;
  }

  // Assign contiguous local indices
  int total_indices = 0;
  for (auto &kv : ispec_to_stacey_faces) {
    boundary_index_mapping(kv.first) = total_indices;
    ++total_indices;
  }

  if (total_indices != total_stacey_elements) {
    KOKKOS_ABORT_WITH_LOCATION("Error: Mismatch in total Stacey surface elements");
  }

  // Verify contiguous mapping
  for (int ispec = 1; ispec < nspec; ++ispec) {
    if ((boundary_index_mapping(ispec) == -1) ||
        (boundary_index_mapping(ispec - 1) == -1))
      continue;
    if (boundary_index_mapping(ispec) != boundary_index_mapping(ispec - 1) + 1)
      throw std::runtime_error("Boundary index mapping is not contiguous");
  }

  // Allocate views
  this->quadrature_point_boundary_tag =
      BoundaryTagView("specfem::assembly::impl::boundaries::stacey::"
                      "quadrature_point_boundary_tag",
                      total_stacey_elements, ngllz, nglly, ngllx);
  this->h_quadrature_point_boundary_tag =
      Kokkos::create_mirror_view(quadrature_point_boundary_tag);

  this->face_weight =
      FaceWeightView("specfem::assembly::impl::boundaries::stacey::face_weight",
                     total_stacey_elements, ngllz, nglly, ngllx);
  this->face_normal =
      FaceNormalView("specfem::assembly::impl::boundaries::stacey::face_normal",
                     total_stacey_elements, ngllz, nglly, ngllx, 3);

  this->h_face_weight = Kokkos::create_mirror_view(face_weight);
  this->h_face_normal = Kokkos::create_mirror_view(face_normal);

  // Populate boundary values
  for (auto &kv : ispec_to_stacey_faces) {
    const int ispec_compute = kv.first;
    const int local_index = boundary_index_mapping(ispec_compute);

    element_boundary_tags[ispec_compute] +=
        specfem::element::boundary_tag::stacey;

    for (int iface : kv.second) {
      const auto face_type = mesh.boundaries.face_type(iface);

      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix) {
            if (!is_on_boundary(face_type, iz, iy, ix, ngllz, nglly, ngllx))
              continue;

            this->h_quadrature_point_boundary_tag(local_index, iz, iy, ix) +=
                specfem::element::boundary_tag::stacey;

            // Compute face normal and weight using Jacobian matrix
            const std::array<type_real, 3> weights = { mesh_assembly.h_weights(iz),
                                                       mesh_assembly.h_weights(iy),
                                                       mesh_assembly.h_weights(ix) };
            specfem::point::index<dimension_tag> point_index(ispec_compute, iz,
                                                              iy, ix);
            specfem::point::jacobian_matrix<dimension_tag, true, false>
                point_jacobian_matrix;
            specfem::assembly::load_on_host(point_index, jacobian_matrix,
                                            point_jacobian_matrix);

            auto [fnormal, fweight] =
                get_boundary_face_and_weight(face_type, weights,
                                             point_jacobian_matrix);

            this->h_face_weight(local_index, iz, iy, ix) = fweight;
            this->h_face_normal(local_index, iz, iy, ix, 0) = fnormal[0];
            this->h_face_normal(local_index, iz, iy, ix, 1) = fnormal[1];
            this->h_face_normal(local_index, iz, iy, ix, 2) = fnormal[2];
          }
        }
      }
    }
  }

  Kokkos::deep_copy(quadrature_point_boundary_tag,
                    h_quadrature_point_boundary_tag);
  Kokkos::deep_copy(face_weight, h_face_weight);
  Kokkos::deep_copy(face_normal, h_face_normal);
}
