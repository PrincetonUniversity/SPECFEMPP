#ifndef _COUPLED_INTERFACE_TPP
#define _COUPLED_INTERFACE_TPP

#include "compute/interface.hpp"
#include "coupled_interface.hpp"
#include "impl/edge/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

template <class self_domain_type, class coupled_domain_type>
specfem::coupled_interface::
    coupled_interface<self_domain_type, coupled_domain_type>::coupled_interface(
        self_domain_type &self_domain, coupled_domain_type &coupled_domain,
        const specfem::compute::coupled_interfaces::coupled_interfaces
            &coupled_interfaces,
        const quadrature_points_type &quadrature_points,
        const specfem::compute::partial_derivatives &partial_derivatives,
        const specfem::kokkos::DeviceView3d<int> ibool,
        const specfem::kokkos::DeviceView1d<type_real> wxgll,
        const specfem::kokkos::DeviceView1d<type_real> wzgll)
    : nedges(coupled_interfaces.elastic_acoustic.num_interfaces),
      self_domain(self_domain), coupled_domain(coupled_domain),
      quadrature_points(quadrature_points),
      edge(self_domain, coupled_domain, quadrature_points, coupled_interfaces,
           partial_derivatives, wxgll, wzgll, ibool) {

  static_assert(std::is_same_v<self_medium, coupled_medium> == false,
                "Error: self_medium cannot be equal to coupled_medium");

  bool constexpr elastic_acoustic_condition =
      (std::is_same_v<self_medium, specfem::enums::element::medium::elastic> &&
       std::is_same_v<coupled_medium,
                      specfem::enums::element::medium::acoustic>) ||
      (std::is_same_v<self_medium, specfem::enums::element::medium::acoustic> &&
       std::is_same_v<coupled_medium,
                      specfem::enums::element::medium::elastic>);

  static_assert(elastic_acoustic_condition,
                "Only acoustic-elastic coupling is supported at the moment.");

  if constexpr (std::is_same_v<self_medium,
                               specfem::enums::element::medium::elastic>) {
    this->self_edge = coupled_interfaces.elastic_acoustic.elastic_edge;
    this->coupled_edge = coupled_interfaces.elastic_acoustic.acoustic_edge;
  } else {
    this->self_edge = coupled_interfaces.elastic_acoustic.acoustic_edge;
    this->coupled_edge = coupled_interfaces.elastic_acoustic.elastic_edge;
  }

  return;
}

template <class self_domain_type, class coupled_domain_type>
void specfem::coupled_interface::coupled_interface<
    self_domain_type, coupled_domain_type>::compute_coupling() {

  if (this->nedges == 0)
    return;

  Kokkos::parallel_for(
      "specfem::coupled_interfaces::coupled_interfaces::compute_coupling",
      specfem::kokkos::DeviceTeam(this->nedges, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        // Get number of quadrature points
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        int iedge_l = team_member.league_rank();
        // Get the edge
        const auto self_edge_l = this->self_edge(iedge_l);
        const auto coupled_edge_l = this->coupled_edge(iedge_l);

        auto npoints = specfem::compute::coupled_interfaces::iterator::npoints(
            self_edge_l, ngllx, ngllz);

        // Iterate over the edges using TeamThreadRange
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, npoints),
            [=](const int ipoint) { edge.compute_coupling(iedge_l, ipoint); });
      });

  Kokkos::fence();

  return;
}

#endif // _COUPLED_INTERFACE_TPP
