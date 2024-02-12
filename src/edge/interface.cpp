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
    const int &ipoint, const specfem::edge::interface &interface, int &i,
    int &j) {
  switch (interface.type) {
  case specfem::enums::edge::type::BOTTOM:
    i = ipoint;
    j = 0;
    break;
  case specfem::enums::edge::type::TOP:
    i = interface.ngll - 1 - ipoint;
    j = interface.ngll - 1;
    break;
  case specfem::enums::edge::type::LEFT:
    i = 0;
    j = ipoint;
    break;
  case specfem::enums::edge::type::RIGHT:
    i = interface.ngll - 1;
    j = interface.ngll - 1 - ipoint;
    break;
  default:
    assert(false && "Invalid edge type");
  }
}

KOKKOS_FUNCTION
void specfem::edge::locate_point_on_coupled_edge(
    const int &ipoint, const specfem::edge::interface &interface, int &i,
    int &j) {
  switch (interface.type) {
  case specfem::enums::edge::type::BOTTOM:
    i = interface.ngll - 1 - ipoint;
    j = 0;
    break;
  case specfem::enums::edge::type::TOP:
    i = ipoint;
    j = interface.ngll - 1;
    break;
  case specfem::enums::edge::type::LEFT:
    i = interface.ngll - 1;
    j = interface.ngll - 1 - ipoint;
    break;
  case specfem::enums::edge::type::RIGHT:
    i = 0;
    j = ipoint;
    break;
  default:
    assert(false && "Invalid edge type");
  }
}
