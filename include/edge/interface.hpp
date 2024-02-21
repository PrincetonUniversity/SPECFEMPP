#ifndef _EDGE_INTERFACE_HPP
#define _EDGE_INTERFACE_HPP

#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace edge {

struct interface {
  specfem::enums::edge::type type;
  int ngll;

  interface() = default;

  interface(const specfem::enums::edge::type type, const int ngll)
      : type(type), ngll(ngll) {}
};

KOKKOS_FUNCTION
int num_points_on_interface(const specfem::edge::interface &interface);

KOKKOS_FUNCTION
void locate_point_on_self_edge(const int &ipoint,
                               const specfem::edge::interface &edge, int &iz,
                               int &ix);

KOKKOS_FUNCTION
void locate_point_on_coupled_edge(const int &ipoint,
                                  const specfem::edge::interface &edge, int &iz,
                                  int &ix);
} // namespace edge
} // namespace specfem

#endif
