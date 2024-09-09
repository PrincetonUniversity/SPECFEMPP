#pragma once

#include "impl/compute_coupling.hpp"
#include "parallel_configuration/edge_config.hpp"
#include "policies/edge.hpp"
#include <Kokkos_Core.hpp>

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag SelfMedium,
          specfem::element::medium_tag CoupledMedium>
specfem::coupled_interface::coupled_interface<WavefieldType, DimensionType,
                                              SelfMedium, CoupledMedium>::
    coupled_interface(const specfem::compute::assembly &assembly) {

  const auto coupled_interfaces = assembly.coupled_interfaces;
  const auto interface_container =
      coupled_interfaces.get_interface_container<SelfMedium, CoupledMedium>();
  const auto field = assembly.fields.get_simulation_field<WavefieldType>();

  this->nedges = interface_container.num_interfaces;
  this->npoints = interface_container.num_points;
  this->interface_data = interface_container;
  this->field = field;
}

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag SelfMedium,
          specfem::element::medium_tag CoupledMedium>
void specfem::coupled_interface::coupled_interface<
    WavefieldType, DimensionType, SelfMedium,
    CoupledMedium>::compute_coupling() {

  if (this->nedges == 0)
    return;

  using ParallelConfig = specfem::parallel_config::default_edge_config<
      DimensionType, Kokkos::DefaultExecutionSpace>;

  using EdgePolicyType = specfem::policy::element_edge<ParallelConfig>;

  const auto edge_factor = this->interface_data.get_edge_factor();
  const auto edge_normal = this->interface_data.get_edge_normal();

  const auto [self_index_mapping, coupled_index_mapping] =
      this->interface_data.get_index_mapping();

  const auto [self_edge_type, coupled_edge_type] =
      this->interface_data.get_edge_type();

  EdgePolicyType edge_policy(self_index_mapping, coupled_index_mapping,
                                   self_edge_type, coupled_edge_type,
                                   this->npoints);

  Kokkos::parallel_for(
      "specfem::coupled_interfaces::compute_coupling",
      static_cast<typename EdgePolicyType::policy_type &>(edge_policy),
      KOKKOS_LAMBDA(const typename EdgePolicyType::member_type &team_member) {
        const auto iterator =
            edge_policy.league_iterator(team_member.league_rank());

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, iterator.edge_size()),
            [=](const int ipoint) {
              const auto index = iterator(ipoint);
              const auto self_index = index.self_index;
              const auto coupled_index = index.coupled_index;
              const int iedge = index.iedge;

              const auto factor = edge_factor(iedge, ipoint);
              const specfem::datatype::ScalarPointViewType<type_real, 2, false>
                  normal(edge_normal(0, iedge, ipoint),
                         edge_normal(1, iedge, ipoint));

              CoupledPointFieldType coupled_field;
              specfem::compute::load_on_device(coupled_index, field,
                                               coupled_field);

              SelfPointFieldType acceleration;
              specfem::coupled_interface::impl::compute_coupling(
                  factor, normal, coupled_field, acceleration);

              specfem::compute::atomic_add_on_device(self_index, acceleration,
                                                     field);
            });
      });
}
