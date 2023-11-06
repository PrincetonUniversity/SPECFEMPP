#include "compute/compute_boundaries.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include <Kokkos_Core.hpp>

specfem::compute::acoustic_free_surface::acoustic_free_surface(
    const specfem::mesh::boundaries::acoustic_free_surface
        &acoustic_free_surface)
    : nelem_acoustic_surface(acoustic_free_surface.nelem_acoustic_surface),
      h_ispec_acoustic_surface(acoustic_free_surface.ispec_acoustic_surface),
      h_type(acoustic_free_surface.type) {

  if (nelem_acoustic_surface > 0) {
    ispec_acoustic_surface = specfem::kokkos::DeviceView1d<int>(
        "specfem::compute::boundaries::acoustic_free_surface::ispec_acoustic_"
        "surface",
        nelem_acoustic_surface);
    type = specfem::kokkos::DeviceView1d<specfem::enums::boundaries::type>(
        "specfem::compute::boundaries::acoustic_free_surface::type",
        nelem_acoustic_surface);
    Kokkos::deep_copy(ispec_acoustic_surface, h_ispec_acoustic_surface);
    Kokkos::deep_copy(type, h_type);
  }
}

KOKKOS_FUNCTION
bool specfem::compute::access::is_on_boundary(
    const specfem::enums::boundaries::type &type, const int &iz, const int &ix,
    const int &ngllz, const int &ngllx) {
  if (type == specfem::enums::boundaries::type::TOP) {
    return iz == ngllz - 1;
  } else if (type == specfem::enums::boundaries::type::BOTTOM) {
    return iz == 0;
  } else if (type == specfem::enums::boundaries::type::LEFT) {
    return ix == 0;
  } else if (type == specfem::enums::boundaries::type::RIGHT) {
    return ix == ngllx - 1;
  } else if (type == specfem::enums::boundaries::type::BOTTOM_RIGHT) {
    return iz == 0 && ix == ngllx - 1;
  } else if (type == specfem::enums::boundaries::type::BOTTOM_LEFT) {
    return iz == 0 && ix == 0;
  } else if (type == specfem::enums::boundaries::type::TOP_RIGHT) {
    return iz == ngllz - 1 && ix == ngllx - 1;
  } else if (type == specfem::enums::boundaries::type::TOP_LEFT) {
    return iz == ngllz - 1 && ix == 0;
  } else {
    return false;
  }
}
