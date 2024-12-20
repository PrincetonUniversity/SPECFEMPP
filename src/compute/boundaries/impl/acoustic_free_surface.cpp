
#include "compute/boundaries/impl/acoustic_free_surface.hpp"
#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {
bool is_on_boundary(specfem::enums::boundaries::type type, int iz, int ix,
                    int ngllz, int ngllx) {
  return (type == specfem::enums::boundaries::type::TOP && iz == ngllz - 1) ||
         (type == specfem::enums::boundaries::type::BOTTOM && iz == 0) ||
         (type == specfem::enums::boundaries::type::LEFT && ix == 0) ||
         (type == specfem::enums::boundaries::type::RIGHT && ix == ngllx - 1) ||
         (type == specfem::enums::boundaries::type::BOTTOM_RIGHT && iz == 0 &&
          ix == ngllx - 1) ||
         (type == specfem::enums::boundaries::type::BOTTOM_LEFT && iz == 0 &&
          ix == 0) ||
         (type == specfem::enums::boundaries::type::TOP_RIGHT &&
          iz == ngllz - 1 && ix == ngllx - 1) ||
         (type == specfem::enums::boundaries::type::TOP_LEFT &&
          iz == ngllz - 1 && ix == 0);
}
} // namespace

specfem::compute::impl::boundaries::acoustic_free_surface::
    acoustic_free_surface(
        const int nspec, const int ngllz, const int ngllx,
        const specfem::mesh::acoustic_free_surface<
            specfem::dimension::type::dim2> &acoustic_free_surface,
        const specfem::compute::mesh_to_compute_mapping &mapping,
        const specfem::compute::properties &properties,
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
    const int ispec_compute = mapping.mesh_to_compute(ispec_mesh);
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
      BoundaryTagView("specfem::compute::impl::boundaries::"
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

  // // ------------------- Sort ispec_acoustic_surface -------------------
  // // There might be better way of doing this but for now I am sorting
  // const int nelements = acoustic_free_surface.nelem_acoustic_surface;
  // std::vector<int> sorted_ispec(nelements);
  // std::vector<specfem::enums::boundaries::type> sorted_type(nelements);

  // std::vector<std::size_t> iota(nelements);
  // std::iota(iota.begin(), iota.end(), 0);

  // // Sort indices based on ispec_acoustic_surface
  // std::sort(iota.begin(), iota.end(), [&](std::size_t i1, std::size_t i2) {
  //   return acoustic_free_surface.ispec_acoustic_surface(i1) <
  //          acoustic_free_surface.ispec_acoustic_surface(i2);
  // });

  // // Reorder ispec_acoustic_surface and type
  // for (int i = 0; i < nelements; ++i) {
  //   sorted_ispec[i] = acoustic_free_surface.ispec_acoustic_surface(iota[i]);
  //   sorted_type[i] = acoustic_free_surface.type(iota[i]);
  // }
  // // -------------------------------------------------------------------

  // // Initialize all index mappings to -1
  // for (int ispec = 0; ispec < nspec; ++ispec) {
  //   boundary_index_mapping(ispec) = -1;
  // }

  // // Assign boundary index mapping
  // int total_indices = 0;
  // for (int i = 0; i < nelements; ++i) {
  //   const int ispec = sorted_ispec[i];
  //   const int ispec_compute = mapping.mesh_to_compute(ispec);
  //   if (boundary_index_mapping(ispec_compute) == -1) {
  //     boundary_index_mapping(ispec_compute) = total_indices;
  //     ++total_indices;
  //   }
  // }

  // // Make sure the index mapping is contiguous
  // for (int ispec = 0; ispec < nspec; ++ispec) {
  //   if (ispec == 0)
  //     continue;

  //   if ((boundary_index_mapping(ispec) == -1) &&
  //       (boundary_index_mapping(ispec - 1) != -1)) {
  //     std::cout << "ispec: " << ispec << std::endl;
  //     std::cout << "boundary_index_mapping(ispec): "
  //               << boundary_index_mapping(ispec) << std::endl;
  //     std::cout << "boundary_index_mapping(ispec - 1): "
  //               << boundary_index_mapping(ispec - 1) << std::endl;
  //     throw std::runtime_error("Boundary index mapping is not contiguous");
  //   } else {
  //     continue;
  //   }

  //   if (boundary_index_mapping(ispec) !=
  //       boundary_index_mapping(ispec - 1) + 1) {
  //     std::cout << "ispec: " << ispec << std::endl;
  //     std::cout << "boundary_index_mapping(ispec): "
  //               << boundary_index_mapping(ispec) << std::endl;
  //     throw std::runtime_error("Boundary index mapping is not contiguous");
  //   }
  // }

  // this->quadrature_point_boundary_tag =
  //     BoundaryTagView("specfem::compute::impl::boundaries::"
  //                     "acoustic_free_surface::quadrature_point_boundary_tag",
  //                     total_indices, ngllz, ngllx);

  // this->h_quadrature_point_boundary_tag =
  //     Kokkos::create_mirror_view(quadrature_point_boundary_tag);

  // // Initialize boundary tags
  // for (int i = 0; i < nelements; ++i) {
  //   const int ispec = sorted_ispec[i];
  //   const int ispec_compute = mapping.mesh_to_compute(ispec);
  //   const auto type = sorted_type[i];
  //   const int index = boundary_index_mapping(ispec_compute);
  //   // Check acoustic free surface element is of acoustic type
  //   if (properties.h_medium_tags(ispec_compute) !=
  //       specfem::element::medium_tag::acoustic) {
  //     throw std::runtime_error(
  //         "Acoustic free surface element is not of acoustic type");
  //   }

  //   element_boundary_tags(ispec_compute) +=
  //       specfem::element::boundary_tag::acoustic_free_surface;

  //   // Assign boundary tag to each quadrature point
  //   for (int iz = 0; iz < ngllz; ++iz) {
  //     for (int ix = 0; ix < ngllx; ++ix) {
  //       if (is_on_boundary(type, iz, ix, ngllz, ngllx)) {
  //         quadrature_point_boundary_tag(index, iz, ix) +=
  //             specfem::element::boundary_tag::acoustic_free_surface;
  //       }
  //     }
  //   }
  // }

  Kokkos::deep_copy(quadrature_point_boundary_tag,
                    h_quadrature_point_boundary_tag);
}
