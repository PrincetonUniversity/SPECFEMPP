#ifndef _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_TPP
#define _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_TPP

#include "compute/interface.hpp"
#include "coupled_interface/impl/edge/edge.hpp"
#include "coupled_interface/impl/edge/elastic_acoustic/elastic_acoustic.hpp"
#include "domain/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <typename qp_type>
specfem::coupled_interface::impl::edges::edge<
    specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>,
    specfem::domain::domain<specfem::enums::element::medium::acoustic,
                            qp_type> >::
    edge(
        const int &inum_edge,
        const specfem::domain::domain<specfem::enums::element::medium::elastic,
                                      qp_type> &self_domain,
        const specfem::domain::domain<specfem::enums::element::medium::acoustic,
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
      coupled_interfaces.elastic_acoustic.elastic_ispec(inum_edge);
  const int coupled_ispec =
      coupled_interfaces.elastic_acoustic.acoustic_ispec(inum_edge);

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

  self_edge_type = coupled_interfaces.elastic_acoustic.elastic_edge(inum_edge);
  coupled_edge_type =
      coupled_interfaces.elastic_acoustic.acoustic_edge(inum_edge);

  self_iterator = specfem::coupled_interface::impl::edges::self_iterator(
      self_edge_type, ngllx, ngllz);
  coupled_iterator = specfem::coupled_interface::impl::edges::coupled_iterator(
      coupled_edge_type, ngllx, ngllz);

  self_field_dot_dot = self_domain.get_field_dot_dot();
  coupled_field = coupled_domain.get_field();

#ifndef NDEBUG
  assert(self_field_dot_dot.extent(1) == self_medium::components);
  assert(coupled_field.extent(1) == coupled_medium::components);
#endif

  return;
}

template <typename qp_type>
void specfem::coupled_interface::impl::edges::edge<
    specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>,
    specfem::domain::domain<specfem::enums::element::medium::acoustic,
                            qp_type> >::compute_coupling(const int &ipoint)
    const {
  int ngllx, ngllz;

  quadrature_points.get_ngll(&ngllx, &ngllz);

  int i, j;
  coupled_iterator(ipoint, i, j);

  int iglob = coupled_ibool(j, i);
  type_real pressure = coupled_field(iglob, 0);

  self_iterator(ipoint, i, j);

  iglob = self_ibool(j, i);

  switch (self_edge_type) {
  case specfem::enums::coupling::edge::type::LEFT:
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 0),
                       -1.0 * wzgll(j) *
                           (xix(j, i) * jacobian(j, i) * pressure));
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 1),
                       -1.0 * wzgll(j) *
                           (xiz(j, i) * jacobian(j, i) * pressure));
    break;
  case specfem::enums::coupling::edge::type::RIGHT:
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 0),
                       wzgll(j) * (xix(j, i) * jacobian(j, i) * pressure));
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 1),
                       wzgll(j) * (xiz(j, i) * jacobian(j, i) * pressure));
    break;
  case specfem::enums::coupling::edge::type::BOTTOM:
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 0),
                       -1.0 * wxgll(i) *
                           (gammax(j, i) * jacobian(j, i) * pressure));
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 1),
                       -1.0 * wxgll(i) *
                           (gammaz(j, i) * jacobian(j, i) * pressure));

    break;
  case specfem::enums::coupling::edge::type::TOP:
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 0),
                       wxgll(i) * (gammax(j, i) * jacobian(j, i) * pressure));
    Kokkos::atomic_add(&self_field_dot_dot(iglob, 1),
                       wxgll(i) * (gammaz(j, i) * jacobian(j, i) * pressure));
    break;
  default:
    assert(false);
    break;
  }
}

#endif // _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_TPP
