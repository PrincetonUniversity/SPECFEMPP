#ifndef _COUPLED_INTERFACE_IMPL_ACOUSTIC_ELASTIC_TPP
#define _COUPLED_INTERFACE_IMPL_ACOUSTIC_ELASTIC_TPP

#include "compute/interface.hpp"
#include "coupled_interface/impl/edge/edge.hpp"
#include "coupled_interface/impl/edge/elastic_acoustic/acoustic_elastic.hpp"
#include "domain/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <typename qp_type>
specfem::coupled_interface::impl::edge<
    specfem::domain::domain<specfem::enums::element::medium::acoustic, qp_type>,
    specfem::domain::domain<specfem::enums::element::medium::elastic,
                            qp_type> >::
    edge(const int &inum_edge,
         const specfem::domain::domain<
             specfem::enums::element::medium::acoustic, qp_type> &self_domain,
         const specfem::domain::domain<specfem::enums::element::medium::elastic,
                                       qp_type> &coupled_domain,
         const qp_type &quadrature_points,
         const specfem::compute::coupled_interfaces::coupled_interfaces
             &coupled_interfaces,
         const specfem::compute::partial_derivatives &partial_derivatives,
         const specfem::kokkos::DeviceView1d<type_real> wxgll,
         const specfem::kokkos::DeviceView1d<type_real> wzgll,
         const specfem::kokkos::DeviceView3d<int> ibool)
    : quadrature_points(quadrature_points), wxgll(wxgll), wzgll(wzgll) {

  int ngllx, ngllz;
  quadrature_points.get_ngll(&ngllx, &ngllz);

#ifndef NDEBUG
  assert(ibool.extent(1) == ngllz);
  assert(ibool.extent(2) == ngllx);
  assert(partial_derivatives.xix.extent(1) == ngllz);
  assert(partial_derivatives.xix.extent(2) == ngllx);
  assert(partial_derivatives.xiz.extent(1) == ngllz);
  assert(partial_derivatives.xiz.extent(2) == ngllx);
  assert(partial_derivatives.gammax.extent(1) == ngllz);
  assert(partial_derivatives.gammax.extent(2) == ngllx);
  assert(partial_derivatives.gammaz.extent(1) == ngllz);
  assert(partial_derivatives.gammaz.extent(2) == ngllx);
  assert(partial_derivatives.jacobian.extent(1) == ngllz);
  assert(partial_derivatives.jacobian.extent(2) == ngllx);
  assert(wxgll.extent(0) == ngllx);
  assert(wzgll.extent(0) == ngllz);
#endif

  const int self_ispec =
      coupled_interfaces.elastic_acoustic.acoustic_ispec(inum_edge);
  const int coupled_ispec =
      coupled_interfaces.elastic_acoustic.elastic_ispec(inum_edge);

  self_ibool = Kokkos::subview(ibool, self_ispec, Kokkos::ALL, Kokkos::ALL);
  coupled_ibool =
      Kokkos::subview(ibool, coupled_ispec, Kokkos::ALL, Kokkos::ALL);

  xix = Kokkos::subview(partial_derivatives.xix, self_ispec, Kokkos::ALL,
                        Kokkos::ALL);
  xiz = Kokkos::subview(partial_derivatives.xiz, self_ispec, Kokkos::ALL,
                        Kokkos::ALL);
  gammax = Kokkos::subview(partial_derivatives.gammax, self_ispec, Kokkos::ALL,
                           Kokkos::ALL);
  gammaz = Kokkos::subview(partial_derivatives.gammaz, self_ispec, Kokkos::ALL,
                           Kokkos::ALL);
  jacobian = Kokkos::subview(partial_derivatives.jacobian, self_ispec,
                             Kokkos::ALL, Kokkos::ALL);

  self_edge_type = coupled_interfaces.elastic_acoustic.acoustic_edge(inum_edge);
  coupled_edge_type =
      coupled_interfaces.elastic_acoustic.elastic_edge(inum_edge);

  self_field_dot_dot = self_domain.get_field_dot_dot();
  coupled_field = coupled_domain.get_field();

#ifndef NDEBUG
  assert(self_field_dot_dot.extent(1) == self_medium::components);
  assert(coupled_field.extent(1) == coupled_medium::components);
#endif

  return;
}

template <typename qp_type>
void specfem::coupled_interface::impl::edge<
    specfem::domain::domain<specfem::enums::element::medium::acoustic, qp_type>,
    specfem::domain::domain<specfem::enums::element::medium::elastic,
                            qp_type> >::compute_coupling(const int &ipoint)
    const {

  int ngllx, ngllz;
  quadrature_points.get_ngll(&ngllx, &ngllz);

  int i, j;
  coupled_iterator(ipoint, coupled_edge_type, ngllx, ngllz, i, j);

  int iglob = coupled_ibool(j, i);
  const type_real displ_x = coupled_field(iglob, 0);
  const type_real displ_z = coupled_field(iglob, 1);

  self_iterator(ipoint, self_edge_type, ngllx, ngllz, i, j);

  iglob = self_ibool(j, i);

  switch (self_edge_type) {
  case specfem::enums::coupling::edge::type::LEFT:
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 0),
                       -1.0 * wzgll(j) *
                           (xix(j, i) * jacobian(j, i) * displ_x +
                            xiz(j, i) * jacobian(j, i) * displ_z));
    break;
  case specfem::enums::coupling::edge::type::RIGHT:
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 0),
                       wzgll(j) * (xix(j, i) * jacobian(j, i) * displ_x +
                                   xiz(j, i) * jacobian(j, i) * displ_z));
    break;
  case specfem::enums::coupling::edge::type::BOTTOM:
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 0),
                       -1.0 * wxgll(i) *
                           (gammax(j, i) * jacobian(j, i) * displ_x +
                            gammaz(j, i) * jacobian(j, i) * displ_z));

    break;
  case specfem::enums::coupling::edge::type::TOP:
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 0),
                       wxgll(i) * (gammax(j, i) * jacobian(j, i) * displ_x +
                                   gammaz(j, i) * jacobian(j, i) * displ_z));
    break;
  default:
    break;
  }

  return;
}

#endif // _COUPLED_INTERFACE_IMPL_ACOUSTIC_ELASTIC_TPP
