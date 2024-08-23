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

  //   Kokkos::parallel_for(
  //       "specfem::coupled_interfaces::compute_coupling",
  //       Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IterateLeft>(
  //           { 0, 0 }, { this->nedges, this->npoints }),
  //       KOKKOS_LAMBDA(const int iedge, const int ipoint) {
  //         const auto self_index = self_index_mapping(iedge, ipoint);
  //         const auto coupled_index = coupled_index_mapping(iedge, ipoint);

  //         const auto factor = edge_factor(iedge, ipoint);
  //         const auto normal = edge_normal(iedge, ipoint);

  //         CoupledPointFieldType coupled_field;
  //         specfem::compute::load_on_device(coupled_index, field,
  //         coupled_field);

  //         SelfPointFieldType acceleration;
  //         specfem::coupled_interface::impl::compute_coupling(
  //             factor, normal, coupled_field, acceleration);

  //         specfem::compute::atomic_add_on_device(self_index, acceleration,
  //         field);
  //       });
}

// #ifndef _COUPLED_INTERFACE_TPP
// #define _COUPLED_INTERFACE_TPP

// #include "compute/interface.hpp"
// #include "coupled_interface.hpp"
// #include "enumerations/interface.hpp"
// #include "impl/edge/interface.hpp"
// #include "kokkos_abstractions.h"
// #include "macros.hpp"
// #include <Kokkos_Core.hpp>

// template <specfem::wavefield::type WavefieldType,
//           specfem::dimension::type DimensionType,
//           specfem::element::medium_tag SelfMedium,
//           specfem::element::medium_tag CoupledMedium>
// specfem::coupled_interface::coupled_interface<WavefieldType, DimensionType,
//                                               SelfMedium, CoupledMedium>::
//     coupled_interface(const specfem::compute::assembly &assembly)
//     : nedges(assembly.coupled_interfaces
//                  .get_interface_container<SelfMedium, CoupledMedium>()
//                  .num_interfaces),
//       interface_data(assembly.coupled_interfaces
//                          .get_interface_container<SelfMedium,
//                          CoupledMedium>()),
//       points(assembly.mesh.points), quadrature(assembly.mesh.quadratures),
//       partial_derivatives(assembly.partial_derivatives),
//       field(assembly.fields.get_simulation_field<WavefieldType>()),
//       edge(assembly) {}

// template <specfem::wavefield::type WavefieldType,
//           specfem::dimension::type DimensionType,
//           specfem::element::medium_tag SelfMedium,
//           specfem::element::medium_tag CoupledMedium>
// void specfem::coupled_interface::coupled_interface<
//     WavefieldType, DimensionType, SelfMedium,
//     CoupledMedium>::compute_coupling() {

//   constexpr bool using_simd = false;

//   if (this->nedges == 0)
//     return;

//   const auto wgll = quadrature.gll.weights;
//   const auto index_mapping = points.index_mapping;

//   Kokkos::parallel_for(
//       "specfem::coupled_interfaces::coupled_interfaces::compute_coupling",
//       specfem::kokkos::DeviceTeam(this->nedges, Kokkos::AUTO, 1),
//       KOKKOS_CLASS_LAMBDA(
//           const specfem::kokkos::DeviceTeam::member_type &team_member) {
//         int iedge_l = team_member.league_rank();

//         // Load the spectral element index and the edge type for the edge
//         //---------------------------------------------------------------------
//         const auto self_edge_type =
//             interface_data.template
//             load_device_edge_type<SelfMedium>(iedge_l);
//         const auto coupled_edge_type =
//             interface_data.template load_device_edge_type<CoupledMedium>(
//                 iedge_l);

//         const int self_index =
//             interface_data.template load_device_index_mapping<SelfMedium>(
//                 iedge_l);
//         const int coupled_index =
//             interface_data.template load_device_index_mapping<CoupledMedium>(
//                 iedge_l);

//         auto npoints =
//         specfem::edge::num_points_on_interface(self_edge_type);
//         //---------------------------------------------------------------------

//         // Iterate over the edges using TeamThreadRange
//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, npoints),
//             [=](const int ipoint) {
//               int iz, ix;
//               specfem::edge::locate_point_on_coupled_edge(
//                   ipoint, coupled_edge_type, iz, ix);

//               specfem::point::index index(coupled_index, iz, ix);

//               // compute normal
//               const auto normal = [&]() {
//                 specfem::point::partial_derivatives2<using_simd, true>
//                 point_derivatives; specfem::compute::load_on_device(index,
//                 partial_derivatives,
//                                                  point_derivatives);
//                 return point_derivatives.compute_normal(coupled_edge_type);
//               }();

//               // get coupling field elements
//               const auto coupled_field_elements =
//                   edge.load_field_elements(index, field);

//               const specfem::datatype::ScalarPointViewType<type_real, 2,
//               using_simd>
//                   weights(wgll(ix), wgll(iz));

//               const auto coupling_terms = edge.compute_coupling_terms(
//                   normal, weights, coupled_edge_type,
//                   coupled_field_elements);

//               specfem::edge::locate_point_on_self_edge(ipoint,
//               self_edge_type,
//                                                        iz, ix);

//               index = specfem::point::index(self_index, iz, ix);

//               // Add coupling contributions
//               specfem::compute::atomic_add_on_device(index, coupling_terms,
//                                                      field);
//             });
//       });

//   Kokkos::fence();

//   return;
// }

// #endif // _COUPLED_INTERFACE_TPP
