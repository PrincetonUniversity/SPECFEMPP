
#include "compute/boundaries/boundaries.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/specfem_enums.hpp"
#include "mesh/boundaries/absorbing_boundaries.hpp"
#include "mesh/boundaries/acoustic_free_surface.hpp"
#include "point/boundary.hpp"
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

// namespace {

// // Every element is tagged with a boundary type
// // An element can be either of the follwing types:
// // 1. Stacey
// // 2. Acoustic Free Surface (Dirichlet)
// // 3. Stacey and Acoustic Free Surface (Composite Stacey-Dirichlet)
// std::tuple<std::vector<specfem::element::boundary_tag_container>,
//            std::vector<specfem::point::boundary> >
// tag_elements(
//     const int nspec, const specfem::compute::properties &properties,
//     const specfem::mesh::boundaries::absorbing_boundary
//     &absorbing_boundaries, const
//     specfem::mesh::boundaries::acoustic_free_surface
//         &acoustic_free_surface) {

//   std::vector<specfem::element::boundary_tag_container> boundary_tag(nspec);
//   std::vector<specfem::point::boundary> boundary_types(nspec);

//   for (int i = 0; i < absorbing_boundaries.nelements; ++i) {
//     const int ispec = absorbing_boundaries.ispec(i);
//     boundary_tag[ispec] += specfem::element::boundary_tag::stacey;
//     boundary_types[ispec].update_boundary(
//         absorbing_boundaries.type(i),
//         specfem::element::boundary_tag::stacey);
//   }

//   for (int i = 0; i < acoustic_free_surface.nelem_acoustic_surface; ++i) {
//     const int ispec = acoustic_free_surface.ispec_acoustic_surface(i);
//     if (properties.h_element_types(ispec) !=
//         specfem::element::medium_tag::acoustic) {
//       throw std::invalid_argument(
//           "Error: Acoustic free surface boundary is not an acoustic
//           element");
//     }
//     boundary_tag[ispec] +=
//         specfem::element::boundary_tag::acoustic_free_surface;
//     boundary_types[ispec].update_boundary(
//         acoustic_free_surface.type(i),
//         specfem::element::boundary_tag::acoustic_free_surface);
//   }

//   return std::make_tuple(boundary_tag, boundary_types);
// }

// // template <typename boundary_type>
// // void assign_boundary(
// //     const std::vector<specfem::element::boundary_tag_container>
// //         boundary_tag,
// //     const std::vector<specfem::compute::access::boundary_types>
// //     boundary_types, const specfem::element::boundary_tag tag,
// //     boundary_type *boundary) {
// //   const int nspec = boundary_tag.size();

// //   boundary->nelements =
// //       std::count(boundary_tag.begin(), boundary_tag.end(), tag);

// //   boundary->ispec = specfem::kokkos::DeviceView1d<int>(
// //       "specfem::compute::boundaries::composite_stacey_dirichlet::ispec",
// //       boundary->nelements);
// //   boundary->type =
// // specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types>(
// // "specfem::compute::boundaries::composite_stacey_dirichlet::type",
// //           boundary->nelements);

// //   boundary->h_ispec = Kokkos::create_mirror_view(boundary->ispec);
// //   boundary->h_type = Kokkos::create_mirror_view(boundary->type);

// //   int index = 0;
// //   for (int ispec = 0; ispec < nspec; ++ispec) {
// //     if (boundary_tag[ispec] == tag) {
// //       boundary->h_ispec(index) = ispec;
// //       boundary->h_type(index) = boundary_types[ispec];
// //       index++;
// //     }
// //   }

// //   assert(index == boundary->nelements);

// //   Kokkos::deep_copy(boundary->ispec, boundary->h_ispec);
// //   Kokkos::deep_copy(boundary->type, boundary->h_type);
// // }
// } // namespace

specfem::compute::boundaries::boundaries(
    const int nspec, const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags &tags,
    const specfem::compute::properties &properties,
    const specfem::mesh::boundaries &boundaries)
    : boundary_tags("specfem::compute::boundaries::boundary_"
                    "tags",
                    nspec),
      h_boundary_tags(Kokkos::create_mirror_view(boundary_tags)),
      boundary_types("specfem::compute::boundaries::boundary_"
                     "types",
                     nspec),
      h_boundary_types(Kokkos::create_mirror_view(boundary_types)) {

  // Tag the elements with the boundary tag and boundary type

  //   for (int i = 0; i < absorbing_boundaries.nelements; ++i) {
  //     const int ispec = absorbing_boundaries.ispec(i);
  //     h_boundary_tags(ispec) += specfem::element::boundary_tag::stacey;
  //     h_boundary_types(ispec).update_boundary(
  //         absorbing_boundaries.type(i),
  //         specfem::element::boundary_tag::stacey);
  //   }

  //   for (int i = 0; i < acoustic_free_surface.nelem_acoustic_surface; ++i) {
  //     const int ispec = acoustic_free_surface.ispec_acoustic_surface(i);
  //     if (properties.h_element_types(ispec) !=
  //         specfem::element::medium_tag::acoustic) {
  //       throw std::invalid_argument(
  //           "Error: Acoustic free surface boundary is not an acoustic
  //           element");
  //     }
  //     h_boundary_tags(ispec) +=
  //         specfem::element::boundary_tag::acoustic_free_surface;
  //     h_boundary_types(ispec).update_boundary(
  //         acoustic_free_surface.type(i),
  //         specfem::element::boundary_tag::acoustic_free_surface);
  //   }

  const auto &absorbing_boundary = boundaries.absorbing_boundary;
  for (int i = 0; i < absorbing_boundary.nelements; ++i) {
    const int ispec = absorbing_boundary.ispec(i);
    const int ispec_compute = mapping.mesh_to_compute(ispec);
    h_boundary_tags(ispec_compute) += specfem::element::boundary_tag::stacey;
    for (int iz = 0; iz < properties.ngllz; ++iz) {
      for (int ix = 0; ix < properties.ngllx; ++ix) {
        if (is_on_boundary(absorbing_boundary.type(i), iz, ix, properties.ngllz,
                           properties.ngllx)) {
          h_boundary_types(ispec_compute, iz, ix) +=
              specfem::element::boundary_tag::stacey;
        }
      }
    }
  }

  const auto &acoustic_free_surface = boundaries.acoustic_free_surface;
  for (int i = 0; i < acoustic_free_surface.nelem_acoustic_surface; ++i) {
    const int ispec = acoustic_free_surface.ispec_acoustic_surface(i);
    const int ispec_compute = mapping.mesh_to_compute(ispec);
    if (properties.h_element_types(ispec_compute) !=
        specfem::element::medium_tag::acoustic) {
      throw std::invalid_argument(
          "Error: Acoustic free surface boundary is not an acoustic element");
    }
    h_boundary_tags(ispec_compute) +=
        specfem::element::boundary_tag::acoustic_free_surface;
    for (int iz = 0; iz < properties.ngllz; ++iz) {
      for (int ix = 0; ix < properties.ngllx; ++ix) {
        if (is_on_boundary(acoustic_free_surface.type(i), iz, ix,
                           properties.ngllz, properties.ngllx)) {
          h_boundary_types(ispec_compute, iz, ix) +=
              specfem::element::boundary_tag::acoustic_free_surface;
        }
      }
    }
  }

  for (int ispec = 0; ispec < nspec; ++ispec) {
    const int ispec_mesh = mapping.compute_to_mesh(ispec);
    if (tags.tags_container(ispec_mesh).boundary_tag !=
        h_boundary_tags(ispec)) {
      throw std::invalid_argument("Error: Boundary tag mismatch between mesh "
                                  "and compute");
    }
  }

  Kokkos::deep_copy(boundary_tags, h_boundary_tags);
  Kokkos::deep_copy(boundary_types, h_boundary_types);

  return;
}

// specfem::compute::acoustic_free_surface::acoustic_free_surface(
//     const specfem::kokkos::HostView1d<int> kmato,
//     const std::vector<std::shared_ptr<specfem::material::material> >
//     &materials, const specfem::mesh::boundaries::absorbing_boundary
//     &absorbing_boundaries, const
//     specfem::mesh::boundaries::acoustic_free_surface
//         &acoustic_free_surface) {
//   const int nspec = kmato.extent(0);
//   if (acoustic_free_surface.nelem_acoustic_surface > 0) {
//     std::vector<specfem::element::boundary_tag_container>
//     boundary_tag(
//         nspec);
//     std::vector<specfem::compute::access::boundary_types>
//     boundary_types(nspec); tag_elements(kmato,
//     specfem::element::medium_tag::acoustic, materials,
//                  absorbing_boundaries, acoustic_free_surface, boundary_tag,
//                  boundary_types);
//     assign_boundary(
//         boundary_tag, boundary_types,
//         specfem::element::boundary_tag::acoustic_free_surface, this);
//   } else {
//     this->nelements = 0;
//   }
// }

// specfem::compute::stacey_medium::stacey_medium(
//     const specfem::element::medium_tag medium,
//     const specfem::kokkos::HostView1d<int> kmato,
//     const std::vector<std::shared_ptr<specfem::material::material> >
//     &materials, const specfem::mesh::boundaries::absorbing_boundary
//     &absorbing_boundaries, const
//     specfem::mesh::boundaries::acoustic_free_surface
//         &acoustic_free_surface) {

//   const int nspec = kmato.extent(0);
//   if (absorbing_boundaries.nelements > 0) {
//     std::vector<specfem::element::boundary_tag_container>
//     boundary_tag(
//         nspec);
//     std::vector<specfem::compute::access::boundary_types>
//     boundary_types(nspec); tag_elements(kmato, medium, materials,
//     absorbing_boundaries,
//                  acoustic_free_surface, boundary_tag, boundary_types);
//     assign_boundary(boundary_tag, boundary_types,
//                     specfem::element::boundary_tag::stacey, this);
//   } else {
//     this->nelements = 0;
//   }
// }

// specfem::compute::stacey::stacey(
//     const specfem::kokkos::HostView1d<int> kmato,
//     const std::vector<std::shared_ptr<specfem::material::material> >
//     &materials, const specfem::mesh::boundaries::absorbing_boundary
//     &absorbing_boundaries, const
//     specfem::mesh::boundaries::acoustic_free_surface
//         &acoustic_free_surface) {
//   this->elastic = specfem::compute::stacey_medium(
//       specfem::element::medium_tag::elastic, kmato, materials,
//       absorbing_boundaries, acoustic_free_surface);
//   this->acoustic = specfem::compute::stacey_medium(
//       specfem::element::medium_tag::acoustic, kmato, materials,
//       absorbing_boundaries, acoustic_free_surface);
//   this->nelements = this->elastic.nelements + this->acoustic.nelements;
// }

// specfem::compute::composite_stacey_dirichlet::composite_stacey_dirichlet(
//     const specfem::kokkos::HostView1d<int> kmato,
//     const std::vector<std::shared_ptr<specfem::material::material> >
//     &materials, const specfem::mesh::boundaries::absorbing_boundary
//     &absorbing_boundaries, const
//     specfem::mesh::boundaries::acoustic_free_surface
//         &acoustic_free_surface) {

//   const int nspec = kmato.extent(0);
//   if (absorbing_boundaries.nelements > 0 &&
//       acoustic_free_surface.nelem_acoustic_surface > 0) {
//     std::vector<specfem::element::boundary_tag_container>
//     boundary_tag(
//         nspec);
//     std::vector<specfem::compute::access::boundary_types>
//     boundary_types(nspec); tag_elements(kmato,
//     specfem::element::medium_tag::acoustic, materials,
//                  absorbing_boundaries, acoustic_free_surface, boundary_tag,
//                  boundary_types);
//     assign_boundary(
//         boundary_tag, boundary_types,
//         specfem::element::boundary_tag::composite_stacey_dirichlet,
//         this);
//   } else {
//     this->nelements = 0;
//   }
// }

// void specfem::compute::access::boundary_types::update_boundary_type(
//     const specfem::enums::boundaries::type &type,
//     const specfem::element::boundary_tag &tag) {
//   if (type == specfem::enums::boundaries::type::TOP) {
//     top = tag;
//   } else if (type == specfem::enums::boundaries::type::BOTTOM) {
//     bottom = tag;
//   } else if (type == specfem::enums::boundaries::type::LEFT) {
//     left = tag;
//   } else if (type == specfem::enums::boundaries::type::RIGHT) {
//     right = tag;
//   } else if (type == specfem::enums::boundaries::type::BOTTOM_RIGHT) {
//     bottom_right = tag;
//   } else if (type == specfem::enums::boundaries::type::BOTTOM_LEFT) {
//     bottom_left = tag;
//   } else if (type == specfem::enums::boundaries::type::TOP_RIGHT) {
//     top_right = tag;
//   } else if (type == specfem::enums::boundaries::type::TOP_LEFT) {
//     top_left = tag;
//   } else {
//     DEVICE_ASSERT(false, "Error: Unknown boundary type");
//   }
// }

// KOKKOS_FUNCTION
// bool specfem::compute::access::is_on_boundary(
//     const specfem::element::boundary_tag &tag,
//     const specfem::compute::access::boundary_types &type, const int &iz,
//     const int &ix, const int &ngllz, const int &ngllx) {

//   return (type.top == tag && iz == ngllz - 1) ||
//          (type.bottom == tag && iz == 0) || (type.left == tag && ix == 0) ||
//          (type.right == tag && ix == ngllx - 1) ||
//          (type.bottom_right == tag && iz == 0 && ix == ngllx - 1) ||
//          (type.bottom_left == tag && iz == 0 && ix == 0) ||
//          (type.top_right == tag && iz == ngllz - 1 && ix == ngllx - 1) ||
//          (type.top_left == tag && iz == ngllz - 1 && ix == 0);
// }
