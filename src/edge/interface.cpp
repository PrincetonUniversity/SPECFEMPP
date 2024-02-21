#include "edge/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>

KOKKOS_FUNCTION
int specfem::edge::num_points_on_interface(
    const specfem::edge::interface &interface) {
  return interface.ngll;
}

KOKKOS_FUNCTION
void specfem::edge::locate_point_on_self_edge(
    const int &ipoint, const specfem::edge::interface &interface, int &iz,
    int &ix) {
  switch (interface.type) {
  case specfem::enums::edge::type::BOTTOM:
    ix = ipoint;
    iz = 0;
    break;
  case specfem::enums::edge::type::TOP:
    ix = interface.ngll - 1 - ipoint;
    iz = interface.ngll - 1;
    break;
  case specfem::enums::edge::type::LEFT:
    ix = 0;
    iz = ipoint;
    break;
  case specfem::enums::edge::type::RIGHT:
    ix = interface.ngll - 1;
    iz = interface.ngll - 1 - ipoint;
    break;
  default:
    assert(false && "Invalid edge type");
  }
}

KOKKOS_FUNCTION
void specfem::edge::locate_point_on_coupled_edge(
    const int &ipoint, const specfem::edge::interface &interface, int &iz,
    int &ix) {
  switch (interface.type) {
  case specfem::enums::edge::type::BOTTOM:
    ix = interface.ngll - 1 - ipoint;
    iz = 0;
    break;
  case specfem::enums::edge::type::TOP:
    ix = ipoint;
    iz = interface.ngll - 1;
    break;
  case specfem::enums::edge::type::LEFT:
    ix = interface.ngll - 1;
    iz = interface.ngll - 1 - ipoint;
    break;
  case specfem::enums::edge::type::RIGHT:
    ix = 0;
    iz = ipoint;
    break;
  default:
    assert(false && "Invalid edge type");
  }
}
