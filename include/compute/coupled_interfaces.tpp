#ifndef _COMPUTE_COUPLED_INTERFACES_TPP
#define _COMPUTE_COUPLED_INTERFACES_TPP

#include "compute/coupled_interfaces.hpp"

template <specfem::compute::coupled_interfaces::iterator::enums::edge
              edge_interface_type>
void specfem::compute::coupled_interfaces::iterator::get_points_along_the_edges(
    const int &ipoint, const specfem::enums::coupling::edge::type &edge,
    const int &ngllx, const int &ngllz, int &i, int &j) {

  bool constexpr self_edge =
      (edge_interface_type ==
       specfem::compute::coupled_interfaces::iterator::enums::edge::self);

  bool constexpr coupled_edge =
      (edge_interface_type ==
       specfem::compute::coupled_interfaces::iterator::enums::edge::coupled);

  static_assert(!(self_edge && coupled_edge),
                "Invalid edge type defined for iterator");

  if constexpr (self_edge) {
    switch (edge) {
    case specfem::enums::coupling::edge::type::BOTTOM:
      i = ipoint;
      j = 0;
      break;
    case specfem::enums::coupling::edge::type::TOP:
      i = ngllx - 1 - ipoint;
      j = ngllz - 1;
      break;
    case specfem::enums::coupling::edge::type::LEFT:
      i = 0;
      j = ipoint;
      break;
    case specfem::enums::coupling::edge::type::RIGHT:
      i = ngllx - 1;
      j = ngllz - 1 - ipoint;
      break;
    default:
      throw std::runtime_error("Invalid edge type");
    }
  } else if constexpr (coupled_edge) {
    switch (edge) {
    case specfem::enums::coupling::edge::type::BOTTOM:
      i = ngllx - 1 - ipoint;
      j = 0;
      break;
    case specfem::enums::coupling::edge::type::TOP:
      i = ipoint;
      j = ngllz - 1;
      break;
    case specfem::enums::coupling::edge::type::LEFT:
      i = ngllx - 1;
      j = ngllz - 1 - ipoint;
      break;
    case specfem::enums::coupling::edge::type::RIGHT:
      i = 0;
      j = ipoint;
      break;
    default:
      throw std::runtime_error("Invalid edge type");
    }
  }

  return;
}

#endif // _COMPUTE_COUPLED_INTERFACES_TPP
