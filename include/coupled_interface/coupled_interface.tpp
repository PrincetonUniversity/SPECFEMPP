#ifndef _COUPLED_INTERFACE_TPP
#define _COUPLED_INTERFACE_TPP

#include "compute/interface.hpp"
#include "coupled_interface.hpp"
#include "impl/edge/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "specfem_enums.hpp"
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
    : self_domain(self_domain), coupled_domain(coupled_domain), quadrature_points(quadrature_points) {

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

  int num_interfaces;

  // Iterate over all domains
  if (elastic_acoustic_condition) {
    // Get the number of edges
    num_interfaces = coupled_interfaces.elastic_acoustic.num_interfaces;
  }

  // Allocate the edges
  edges = specfem::kokkos::DeviceView1d<specfem::coupled_interface::impl::edges::edge<
      self_domain_type, coupled_domain_type> >(
      "specfem::coupled_interfaces::coupled_interfaces::edges", num_interfaces);

  auto h_edges = Kokkos::create_mirror_view(edges);

  // Iterate over the edges
  for (int inum_edge = 0; inum_edge < num_interfaces; ++inum_edge) {
    // Create the edge
    h_edges(inum_edge) =
        specfem::coupled_interface::impl::edges::edge<self_domain_type,
                                               coupled_domain_type>(
            inum_edge, self_domain, coupled_domain, quadrature_points,
            coupled_interfaces, partial_derivatives, wxgll, wzgll, ibool);
  }

  Kokkos::deep_copy(edges, h_edges);

  return;
}

template <class self_domain_type, class coupled_domain_type>
void specfem::coupled_interface::coupled_interface<
    self_domain_type, coupled_domain_type>::compute_coupling() {

  const int nedges = this->edges.extent(0);

  Kokkos::parallel_for(
      "specfem::coupled_interfaces::coupled_interfaces::compute_coupling",
      specfem::kokkos::DeviceTeam(nedges, 5, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        // Get number of quadrature points
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        int iedge = team_member.league_rank();
        // Get the edge
        auto edge = this->edges(iedge);
        // Get the edge types
        specfem::enums::coupling::edge::type self_edge_type;
        specfem::enums::coupling::edge::type coupled_edge_type;
        edge.get_edges(self_edge_type, coupled_edge_type);
        // Get the number of points along the edge
        auto npoints =
            specfem::compute::coupled_interfaces::iterator::npoints(
                self_edge_type, ngllx, ngllz);

        // Iterate over the edges using TeamThreadRange
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, npoints),
            [=](const int ipoint) { edge.compute_coupling(ipoint); });
      });

  Kokkos::fence();

  return;
}

#endif // _COUPLED_INTERFACE_TPP
