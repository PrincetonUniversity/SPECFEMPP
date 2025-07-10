#pragma once

#include <array>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "enumerations/interface.hpp"
#include "macros.hpp"
#include "stacey.hpp"
#include "utilities.hpp"
#include <Kokkos_Core.hpp>

specfem::assembly::boundaries_impl::stacey<specfem::dimension::type::dim2>::
    stacey(const int nspec, const int ngllz, const int ngllx,
           const specfem::mesh::absorbing_boundary<dimension_tag> &stacey,
           const specfem::assembly::mesh<dimension_tag> &mesh,
           const specfem::assembly::jacobian_matrix &jacobian_matrix,
           const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
           std::vector<specfem::element::boundary_tag_container>
               &element_boundary_tags) {

  // We need to make sure that boundary index mapping maps every spectral
  // element index to the corresponding index within
  // quadrature_point_boundary_tag

  // mesh.absorbing_boundary.ispec_absorbing_boundary stores the ispec for
  // every Stacey surface. At the corners of the mesh, multiple surfaces
  // belong to the same ispec. The first part of the code assigns a unique index
  // to each ispec.

  // For SIMD loads we need to ensure that there is a contiguous mapping within
  // ispec and index of boundary_index_mapping i.e. boundary_index_mapping(ispec
  // +1) - boundary_index_mapping(ispec) = 1

  // -------------------------------------------------------------------

  // Create a map from ispec to index in stacey

  const int nelements = stacey.nelements;

  std::map<int, std::vector<int> > ispec_to_stacey;

  for (int i = 0; i < nelements; ++i) {
    const int ispec_mesh = stacey.index_mapping(i);
    const int ispec_compute = mesh.mesh_to_compute(ispec_mesh);
    if (ispec_to_stacey.find(ispec_compute) == ispec_to_stacey.end()) {
      ispec_to_stacey[ispec_compute] = { i };
    } else {
      ispec_to_stacey[ispec_compute].push_back(i);
    }
  }

  const int total_stacey_elements = ispec_to_stacey.size();

  // -------------------------------------------------------------------

  // Assign boundary index mapping

  // Initialize all index mappings to -1
  for (int ispec = 0; ispec < nspec; ++ispec) {
    boundary_index_mapping(ispec) = -1;
  }

  // Assign boundary index mapping
  int total_indices = 0;

  for (auto &map : ispec_to_stacey) {
    const int ispec = map.first;
    boundary_index_mapping(ispec) = total_indices;
    ++total_indices;
  }

  ASSERT(total_indices == total_stacey_elements,
         "Error: Total number of Stacey elements do not match");

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

      std::cout << "ispec: " << ispec << std::endl;
      std::cout << "boundary_index_mapping(ispec): "
                << boundary_index_mapping(ispec) << std::endl;
      std::cout << "boundary_index_mapping(ispec - 1): "
                << boundary_index_mapping(ispec - 1) << std::endl;

      throw std::runtime_error("Boundary index mapping is not contiguous");
    }
  }

  // -------------------------------------------------------------------

  // -------------------------------------------------------------------

  // Initialize views

  this->quadrature_point_boundary_tag =
      BoundaryTagView("specfem::assembly::impl::boundaries::stacey::"
                      "quadrature_point_boundary_tag",
                      total_stacey_elements, ngllz, ngllx);

  this->h_quadrature_point_boundary_tag =
      Kokkos::create_mirror_view(quadrature_point_boundary_tag);

  this->edge_weight = EdgeWeightView("specfem::assembly::impl::boundaries::"
                                     "stacey::edge_weight",
                                     total_stacey_elements, ngllz, ngllx);

  this->edge_normal = EdgeNormalView("specfem::assembly::impl::boundaries::"
                                     "stacey::edge_normal",
                                     total_stacey_elements, ngllz, ngllx, 2);

  this->h_edge_weight = Kokkos::create_mirror_view(edge_weight);
  this->h_edge_normal = Kokkos::create_mirror_view(edge_normal);

  // -------------------------------------------------------------------

  // Assign boundary values

  for (auto &map : ispec_to_stacey) {
    const int ispec_compute = map.first;
    const auto &indices = map.second;
    const int local_index = boundary_index_mapping(ispec_compute);

    element_boundary_tags[ispec_compute] +=
        specfem::element::boundary_tag::stacey;

    for (int i : indices) {
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          if (is_on_boundary(stacey.type(i), iz, ix, ngllz, ngllx)) {
            this->h_quadrature_point_boundary_tag(local_index, iz, ix) +=
                specfem::element::boundary_tag::stacey;

            // Compute edge normal and edge weight
            std::array<type_real, 2> weights = { mesh.h_weights(ix),
                                                 mesh.h_weights(iz) };
            specfem::point::index<dimension_tag> index(ispec_compute, iz, ix);
            specfem::point::jacobian_matrix<dimension_tag, true, false>
                point_jacobian_matrix;
            specfem::assembly::load_on_host(index, jacobian_matrix,
                                            point_jacobian_matrix);

            auto [edge_normal, edge_weight] = get_boundary_edge_and_weight(
                stacey.type(i), weights, point_jacobian_matrix);

            // ------------------- Assign edge normal and edge weight

            this->h_edge_weight(local_index, iz, ix) = edge_weight;
            this->h_edge_normal(local_index, iz, ix, 0) = edge_normal[0];
            this->h_edge_normal(local_index, iz, ix, 1) = edge_normal[1];
          }
        }
      }
    }
  }

  Kokkos::deep_copy(quadrature_point_boundary_tag,
                    h_quadrature_point_boundary_tag);
  Kokkos::deep_copy(edge_weight, h_edge_weight);
  Kokkos::deep_copy(edge_normal, h_edge_normal);
}
