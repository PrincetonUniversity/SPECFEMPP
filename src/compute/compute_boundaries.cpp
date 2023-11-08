#include "compute/compute_boundaries.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "mesh/mesh.hpp"
#include <Kokkos_Core.hpp>

specfem::compute::acoustic_free_surface::acoustic_free_surface(
    const specfem::kokkos::HostView1d<int> kmato,
    std::vector<specfem::material::material *> materials,
    const specfem::mesh::boundaries::acoustic_free_surface
        &acoustic_free_surface) {
  const int nspec = kmato.extent(0);
  if (acoustic_free_surface.nelem_acoustic_surface > 0) {
    std::vector<specfem::enums::element::boundary_tag> boundary_tag(nspec);
    std::vector<specfem::compute::access::boundary_types> boundary_types(nspec);
    // make sure all elements are initialized to none
    std::fill(boundary_tag.begin(), boundary_tag.end(),
              specfem::enums::element::boundary_tag::none);
    // make sure all elements are do not have any boundary
    std::fill(boundary_types.begin(), boundary_types.end(),
              specfem::compute::access::boundary_types());

    for (int i = 0; i < acoustic_free_surface.nelem_acoustic_surface; ++i) {
      const int ispec = acoustic_free_surface.ispec_acoustic_surface(i);
      boundary_tag[ispec] =
          specfem::enums::element::boundary_tag::acoustic_free_surface;
      if (materials[kmato(ispec)]->get_ispec_type() !=
          specfem::enums::element::type::acoustic) {
        throw std::invalid_argument(
            "Error: Acoustic free surface boundary is not an acoustic element");
      }
      boundary_types[ispec].update_boundary_type(acoustic_free_surface.type(i));
    }

    this->nelem_acoustic_surface = std::count(
        boundary_tag.begin(), boundary_tag.end(),
        specfem::enums::element::boundary_tag::acoustic_free_surface);

    std::cout << "Number of acoustic free surface elements: "
              << nelem_acoustic_surface << std::endl;

    this->ispec_acoustic_surface = specfem::kokkos::DeviceView1d<int>(
        "specfem::compute::boundaries::acoustic_free_surface::ispec_acoustic_"
        "surface",
        nelem_acoustic_surface);
    this->type =
        specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types>(
            "specfem::compute::boundaries::acoustic_free_surface::type",
            nelem_acoustic_surface);

    this->h_ispec_acoustic_surface =
        Kokkos::create_mirror_view(ispec_acoustic_surface);
    this->h_type = Kokkos::create_mirror_view(type);

    int index = 0;
    for (int ispec = 0; ispec < nspec; ++ispec) {
      if (boundary_tag[ispec] ==
          specfem::enums::element::boundary_tag::acoustic_free_surface) {
        this->h_ispec_acoustic_surface(index) = ispec;
        this->h_type(index) = boundary_types[ispec];
        index++;
      }
    }

    assert(index == nelem_acoustic_surface);

    Kokkos::deep_copy(this->ispec_acoustic_surface,
                      this->h_ispec_acoustic_surface);
    Kokkos::deep_copy(this->type, this->h_type);
  } else {
    this->nelem_acoustic_surface = 0;
  }
}

KOKKOS_FUNCTION void
specfem::compute::access::boundary_types::update_boundary_type(
    const specfem::enums::boundaries::type &type) {
  if (type == specfem::enums::boundaries::type::TOP) {
    top = true;
  } else if (type == specfem::enums::boundaries::type::BOTTOM) {
    bottom = true;
  } else if (type == specfem::enums::boundaries::type::LEFT) {
    left = true;
  } else if (type == specfem::enums::boundaries::type::RIGHT) {
    right = true;
  } else if (type == specfem::enums::boundaries::type::BOTTOM_RIGHT) {
    bottom_right = true;
  } else if (type == specfem::enums::boundaries::type::BOTTOM_LEFT) {
    bottom_left = true;
  } else if (type == specfem::enums::boundaries::type::TOP_RIGHT) {
    top_right = true;
  } else if (type == specfem::enums::boundaries::type::TOP_LEFT) {
    top_left = true;
  } else {
    ASSERT(false, "Error: Unknown boundary type");
  }
}

KOKKOS_FUNCTION
bool specfem::compute::access::is_on_boundary(
    const specfem::compute::access::boundary_types &type, const int &iz,
    const int &ix, const int &ngllz, const int &ngllx) {

  return (type.top && iz == ngllz - 1) || (type.bottom && iz == 0) ||
         (type.left && ix == 0) || (type.right && ix == ngllx - 1) ||
         (type.bottom_right && iz == 0 && ix == ngllx - 1) ||
         (type.bottom_left && iz == 0 && ix == 0) ||
         (type.top_right && iz == ngllz - 1 && ix == ngllx - 1) ||
         (type.top_left && iz == ngllz - 1 && ix == 0);
}
