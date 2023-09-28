#ifndef _COUPLED_INTERFACE_IMPL_EDGE_HPP
#define _COUPLED_INTERFACE_IMPL_EDGE_HPP

#include "compute/coupled_interfaces.hpp"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace coupled_interface {
namespace impl {
namespace edges {

struct self_iterator {
  int ngllx;
  int ngllz;

  self_iterator() = default;

  self_iterator(const int &ngllx, const int &ngllz)
      : ngllx(ngllx), ngllz(ngllz){};

  KOKKOS_FUNCTION
  void operator()(const int &ipoint,
                  const specfem::enums::coupling::edge::type &iedge_type,
                  int &i, int &j) const {
    specfem::compute::coupled_interfaces::iterator::self_iterator(
        ipoint, iedge_type, this->ngllx, this->ngllz, i, j);
    return;
  }
};

struct coupled_iterator {
  int ngllx;
  int ngllz;

  coupled_iterator() = default;

  coupled_iterator(const int &ngllx, const int &ngllz)
      : ngllx(ngllx), ngllz(ngllz){};

  KOKKOS_FUNCTION
  void operator()(const int &ipoint,
                  const specfem::enums::coupling::edge::type &iedge_type,
                  int &i, int &j) const {
    specfem::compute::coupled_interfaces::iterator::coupled_iterator(
        ipoint, iedge_type, this->ngllx, this->ngllz, i, j);
    return;
  }
};

template <class self_domain, class coupled_domain> class edge {};
} // namespace edges
} // namespace impl
} // namespace coupled_interface
} // namespace specfem

#endif // _COUPLED_INTERFACE_IMPL_EDGE_HPP