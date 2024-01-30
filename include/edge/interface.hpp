#ifndef _EDGE_INTERFACE_HPP
#define _EDGE_INTERFACE_HPP

#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace edge {

struct interface {
  specfem::enums::edge::type type;

  interface() = default;

  interface(const specfem::enums::edge::type type) : type(type) {}
};

KOKKOS_FUNCTION
int num_points_on_interface(const specfem::edge::interface &interface,
                            const int ngllz, const int ngllx);

KOKKOS_FUNCTION
void locate_point_on_self_edge(const int &ipoint,
                               const specfem::edge::interface &edge,
                               const int ngllx, const int ngllz, int &i,
                               int &j);

KOKKOS_FUNCTION
void locate_point_on_coupled_edge(const int &ipoint,
                                  const specfem::edge::interface &edge,
                                  const int ngllx, const int ngllz, int &i,
                                  int &j);
} // namespace edge
} // namespace specfem

#endif
