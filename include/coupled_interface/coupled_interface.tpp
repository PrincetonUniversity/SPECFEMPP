#ifndef _COUPLED_INTERFACE_TPP
#define _COUPLED_INTERFACE_TPP

#include "compute/interface.hpp"
#include "coupled_interface.hpp"
#include "enumerations/interface.hpp"
#include "impl/edge/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include <Kokkos_Core.hpp>

template <class self_medium, class coupled_medium>
specfem::coupled_interface::coupled_interface<self_medium, coupled_medium>::
    coupled_interface(const specfem::compute::assembly &assembly)
    : nedges(assembly.coupled_interfaces
                 .get_interface_container<self_medium::value,
                                          coupled_medium::value>()
                 .num_interfaces),
      interface_data(assembly.coupled_interfaces.get_interface_container<
                     self_medium::value, coupled_medium::value>()),
      points(assembly.mesh.points), quadrature(assembly.mesh.quadratures),
      partial_derivatives(assembly.partial_derivatives),
      global_index_mapping(assembly.fields.forward.assembly_index_mapping),
      self_field(assembly.fields.forward.get_field<self_medium>()),
      coupled_field(assembly.fields.forward.get_field<coupled_medium>()),
      edge(assembly) {}

template <class self_domain_type, class coupled_domain_type>
void specfem::coupled_interface::coupled_interface<
    self_domain_type, coupled_domain_type>::compute_coupling() {

  if (this->nedges == 0)
    return;

  const auto wgll = quadrature.gll.weights;
  const auto index_mapping = points.index_mapping;

  Kokkos::parallel_for(
      "specfem::coupled_interfaces::coupled_interfaces::compute_coupling",
      specfem::kokkos::DeviceTeam(this->nedges, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int iedge_l = team_member.league_rank();

        // Load the spectral element index and the edge type for the edge
        //---------------------------------------------------------------------
        const auto self_edge_type =
            interface_data
                .template load_device_edge_type<self_medium_type::value>(
                    iedge_l);
        const auto coupled_edge_type =
            interface_data
                .template load_device_edge_type<coupled_medium_type::value>(
                    iedge_l);

        const int self_index =
            interface_data
                .template load_device_index_mapping<self_medium_type::value>(
                    iedge_l);
        const int coupled_index =
            interface_data
                .template load_device_index_mapping<coupled_medium_type::value>(
                    iedge_l);

        auto npoints = specfem::edge::num_points_on_interface(self_edge_type);
        //---------------------------------------------------------------------

        // Iterate over the edges using TeamThreadRange
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, npoints),
            [=](const int ipoint) {
              int iz, ix;
              specfem::edge::locate_point_on_coupled_edge(
                  ipoint, coupled_edge_type, iz, ix);

              // compute normal
              const auto normal =
                  partial_derivatives
                      .load_device_derivatives<true>(coupled_index, iz, ix)
                      .compute_normal(coupled_edge_type);

              // get coupling field elements
              const int coupled_global_index = global_index_mapping(
                  index_mapping(coupled_index, iz, ix),
                  static_cast<int>(coupled_medium_type::value));
              const auto coupled_field_elements =
                  edge.load_field_elements(coupled_global_index, coupled_field);

              const specfem::kokkos::array_type<type_real, 2> weights(wgll(ix),
                                                                      wgll(iz));

              const auto coupling_terms = edge.compute_coupling_terms(
                  normal, weights, coupled_edge_type, coupled_field_elements);

              specfem::edge::locate_point_on_self_edge(ipoint, self_edge_type,
                                                       iz, ix);

              // Add coupling contributions
              const int self_global_index = global_index_mapping(
                  index_mapping(self_index, iz, ix),
                  static_cast<int>(self_medium_type::value));

              Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                for (int i = 0; i < self_medium_type::components; i++) {
                  Kokkos::atomic_add(
                      &self_field.field_dot_dot(self_global_index, i),
                      coupling_terms[i]);
                }
              });
            });
      });

  Kokkos::fence();

  return;
}

#endif // _COUPLED_INTERFACE_TPP
