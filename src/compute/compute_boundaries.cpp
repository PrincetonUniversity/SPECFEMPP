#include "compute/compute_boundaries.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "mesh/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace {

// Every element is tagged with a boundary type
// An element can be either of the follwing types:
// 1. Stacey
// 2. Acoustic Free Surface (Dirichlet)
// 3. Stacey and Acoustic Free Surface (Composite Stacey-Dirichlet)
void tag_elements(
    const specfem::kokkos::HostView1d<int> &kmato,
    const specfem::enums::element::type &medium,
    const std::vector<specfem::material::material *> &materials,
    const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
    const specfem::mesh::boundaries::acoustic_free_surface
        &acoustic_free_surface,
    std::vector<specfem::enums::element::boundary_tag_container> &boundary_tag,
    std::vector<specfem::compute::access::boundary_types> &boundary_types) {
  const int nspec = kmato.extent(0);

  ASSERT(boundary_tag.size() == nspec, "Error: Boundary tag size mismatch");
  ASSERT(boundary_types.size() == nspec, "Error: Boundary types size mismatch");

  for (int i = 0; i < absorbing_boundaries.nelements; ++i) {
    const int ispec = absorbing_boundaries.ispec(i);
    // Only tag acoustic elements
    if (materials[kmato(ispec)]->get_ispec_type() == medium) {
      boundary_tag[ispec] = specfem::enums::element::boundary_tag::stacey;
      boundary_types[ispec].update_boundary_type(
          absorbing_boundaries.type(i),
          specfem::enums::element::boundary_tag::stacey);
    }
  }

  for (int i = 0; i < acoustic_free_surface.nelem_acoustic_surface; ++i) {
    const int ispec = acoustic_free_surface.ispec_acoustic_surface(i);
    if (materials[kmato(ispec)]->get_ispec_type() !=
        specfem::enums::element::type::acoustic) {
      throw std::invalid_argument(
          "Error: Acoustic free surface boundary is not an acoustic element");
    }
    boundary_tag[ispec] =
        specfem::enums::element::boundary_tag::acoustic_free_surface;
    boundary_types[ispec].update_boundary_type(
        acoustic_free_surface.type(i),
        specfem::enums::element::boundary_tag::acoustic_free_surface);
  }
}

template <typename boundary_type, typename tag_type>
void assign_boundary(
    const std::vector<specfem::enums::element::boundary_tag_container>
        boundary_tag,
    const std::vector<specfem::compute::access::boundary_types> boundary_types,
    const tag_type tag, boundary_type *boundary) {
  const int nspec = boundary_tag.size();

  boundary->nelements =
      std::count(boundary_tag.begin(), boundary_tag.end(), tag);

  boundary->ispec = specfem::kokkos::DeviceView1d<int>(
      "specfem::compute::boundaries::composite_stacey_dirichlet::ispec",
      boundary->nelements);
  boundary->type =
      specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types>(
          "specfem::compute::boundaries::composite_stacey_dirichlet::type",
          boundary->nelements);

  boundary->h_ispec = Kokkos::create_mirror_view(boundary->ispec);
  boundary->h_type = Kokkos::create_mirror_view(boundary->type);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    if (boundary_tag[ispec] == tag) {
      boundary->h_ispec(index) = ispec;
      boundary->h_type(index) = boundary_types[ispec];
      index++;
    }
  }

  assert(index == boundary->nelements);

  Kokkos::deep_copy(boundary->ispec, boundary->h_ispec);
  Kokkos::deep_copy(boundary->type, boundary->h_type);
}
} // namespace

specfem::compute::acoustic_free_surface::acoustic_free_surface(
    const specfem::kokkos::HostView1d<int> kmato,
    const std::vector<specfem::material::material *> materials,
    const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
    const specfem::mesh::boundaries::acoustic_free_surface
        &acoustic_free_surface) {
  const int nspec = kmato.extent(0);
  if (acoustic_free_surface.nelem_acoustic_surface > 0) {
    std::vector<specfem::enums::element::boundary_tag_container> boundary_tag(
        nspec);
    std::vector<specfem::compute::access::boundary_types> boundary_types(nspec);
    tag_elements(kmato, specfem::enums::element::type::acoustic, materials,
                 absorbing_boundaries, acoustic_free_surface, boundary_tag,
                 boundary_types);
    assign_boundary(
        boundary_tag, boundary_types,
        specfem::enums::element::boundary_tag::acoustic_free_surface, this);
  } else {
    this->nelements = 0;
  }
}

specfem::compute::stacey_medium::stacey_medium(
    const specfem::enums::element::type medium,
    const specfem::kokkos::HostView1d<int> kmato,
    const std::vector<specfem::material::material *> materials,
    const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
    const specfem::mesh::boundaries::acoustic_free_surface
        &acoustic_free_surface) {

  const int nspec = kmato.extent(0);
  if (absorbing_boundaries.nelements > 0) {
    std::vector<specfem::enums::element::boundary_tag_container> boundary_tag(
        nspec);
    std::vector<specfem::compute::access::boundary_types> boundary_types(nspec);
    tag_elements(kmato, medium, materials, absorbing_boundaries,
                 acoustic_free_surface, boundary_tag, boundary_types);
    assign_boundary(boundary_tag, boundary_types,
                    specfem::enums::element::boundary_tag::stacey, this);
  } else {
    this->nelements = 0;
  }
}

specfem::compute::stacey::stacey(
    const specfem::kokkos::HostView1d<int> kmato,
    const std::vector<specfem::material::material *> materials,
    const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
    const specfem::mesh::boundaries::acoustic_free_surface
        &acoustic_free_surface) {
  this->elastic = specfem::compute::stacey_medium(
      specfem::enums::element::type::elastic, kmato, materials,
      absorbing_boundaries, acoustic_free_surface);
  this->acoustic = specfem::compute::stacey_medium(
      specfem::enums::element::type::acoustic, kmato, materials,
      absorbing_boundaries, acoustic_free_surface);
  this->nelements = this->elastic.nelements + this->acoustic.nelements;
}

specfem::compute::composite_stacey_dirichlet::composite_stacey_dirichlet(
    const specfem::kokkos::HostView1d<int> kmato,
    const std::vector<specfem::material::material *> materials,
    const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
    const specfem::mesh::boundaries::acoustic_free_surface
        &acoustic_free_surface) {

  const int nspec = kmato.extent(0);
  if (absorbing_boundaries.nelements > 0 &&
      acoustic_free_surface.nelem_acoustic_surface > 0) {
    std::vector<specfem::enums::element::boundary_tag_container> boundary_tag(
        nspec);
    std::vector<specfem::compute::access::boundary_types> boundary_types(nspec);
    tag_elements(kmato, specfem::enums::element::type::acoustic, materials,
                 absorbing_boundaries, acoustic_free_surface, boundary_tag,
                 boundary_types);
    assign_boundary(
        boundary_tag, boundary_types,
        std::tuple<specfem::enums::element::boundary_tag,
                   specfem::enums::element::boundary_tag>(
            specfem::enums::element::boundary_tag::stacey,
            specfem::enums::element::boundary_tag::acoustic_free_surface),
        this);
  } else {
    this->nelements = 0;
  }
}

void specfem::compute::access::boundary_types::update_boundary_type(
    const specfem::enums::boundaries::type &type,
    const specfem::enums::element::boundary_tag &tag) {
  if (type == specfem::enums::boundaries::type::TOP) {
    top = tag;
  } else if (type == specfem::enums::boundaries::type::BOTTOM) {
    bottom = tag;
  } else if (type == specfem::enums::boundaries::type::LEFT) {
    left = tag;
  } else if (type == specfem::enums::boundaries::type::RIGHT) {
    right = tag;
  } else if (type == specfem::enums::boundaries::type::BOTTOM_RIGHT) {
    bottom_right = tag;
  } else if (type == specfem::enums::boundaries::type::BOTTOM_LEFT) {
    bottom_left = tag;
  } else if (type == specfem::enums::boundaries::type::TOP_RIGHT) {
    top_right = tag;
  } else if (type == specfem::enums::boundaries::type::TOP_LEFT) {
    top_left = tag;
  } else {
    DEVICE_ASSERT(false, "Error: Unknown boundary type");
  }
}

KOKKOS_FUNCTION
bool specfem::compute::access::is_on_boundary(
    const specfem::enums::element::boundary_tag &tag,
    const specfem::compute::access::boundary_types &type, const int &iz,
    const int &ix, const int &ngllz, const int &ngllx) {

  return (type.top == tag && iz == ngllz - 1) ||
         (type.bottom == tag && iz == 0) || (type.left == tag && ix == 0) ||
         (type.right == tag && ix == ngllx - 1) ||
         (type.bottom_right == tag && iz == 0 && ix == ngllx - 1) ||
         (type.bottom_left == tag && iz == 0 && ix == 0) ||
         (type.top_right == tag && iz == ngllz - 1 && ix == ngllx - 1) ||
         (type.top_left == tag && iz == ngllz - 1 && ix == 0);
}
