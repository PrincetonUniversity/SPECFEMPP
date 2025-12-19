#pragma once

#include "boundary_conditions/boundary_conditions.hpp"
#include "compute_coupling.hpp"
#include "enumerations/connections.hpp"
#include "enumerations/interface.hpp"
#include "execution/chunked_intersection_iterator.hpp"
#include "execution/for_all.hpp"
#include "medium/compute_coupling.hpp"
#include "parallel_configuration/chunk_edge_config.hpp"
#include "specfem/assembly.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include "specfem/point/interface_index.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

#include "algorithms/integrate/coupling_integral1d.hpp"

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_coupling(
    std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::weakly_conforming> /*unused*/,
    const specfem::assembly::assembly<DimensionTag> &assembly) {

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto connection_tag =
      specfem::connections::type::weakly_conforming;
  constexpr static auto interface_tag = InterfaceTag;
  constexpr static auto boundary_tag = BoundaryTag;
  constexpr static auto wavefield = WavefieldType;

  constexpr static auto self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();

  const auto &conforming_interfaces = assembly.conforming_interfaces;
  const auto [self_edges, coupled_edges] =
      assembly.edge_types.get_edges_on_device(connection_tag, interface_tag,
                                              boundary_tag);

  if (self_edges.n_edges != coupled_edges.n_edges) {
    KOKKOS_ABORT_WITH_LOCATION(
        "Mismatch in number of self and coupled edges in compute_coupling.");
  }

  if (self_edges.n_edges == 0 && coupled_edges.n_edges == 0)
    return;

  const auto &field =
      assembly.fields.template get_simulation_field<wavefield>();
  const auto &boundaries = assembly.boundaries;

  const auto num_points = assembly.mesh.element_grid.ngllx;

  using parallel_config = specfem::parallel_configuration::default_chunk_edge_config<
      DimensionTag, Kokkos::DefaultExecutionSpace>;

  using CoupledFieldType = typename specfem::interface::attributes<
      dimension_tag, interface_tag>::template coupled_field_t<connection_tag>;
  using SelfFieldType = typename specfem::interface::attributes<
      dimension_tag, interface_tag>::template self_field_t<connection_tag>;

  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension_tag, false>;

  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_edges, coupled_edges);

  specfem::execution::for_all(
      "specfem::kokkos_kernels::impl::compute_coupling", chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();

        specfem::point::conforming_interface<dimension_tag, interface_tag,
                                             boundary_tag>
            point_interface_data;
        specfem::assembly::load_on_device(
            index.self_index, conforming_interfaces, point_interface_data);

        CoupledFieldType coupled_field;
        specfem::assembly::load_on_device(index.coupled_index, field,
                                          coupled_field);
        SelfFieldType self_field;

        specfem::medium::compute_coupling(point_interface_data, coupled_field,
                                          self_field);

        PointBoundaryType point_boundary;
        specfem::assembly::load_on_device(index.self_index, boundaries,
                                          point_boundary);
        if constexpr (BoundaryTag ==
                      specfem::element::boundary_tag::acoustic_free_surface) {
          specfem::boundary_conditions::apply_boundary_conditions(
              point_boundary, self_field);
        }

        specfem::assembly::atomic_add_on_device(index.self_index, field,
                                                self_field);
      });

  return;
}

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          int NQuad_intersection,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_coupling(
    std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::nonconforming> /*unused*/,
    const specfem::assembly::assembly<DimensionTag> &assembly) {

  constexpr bool using_simd = false;

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto connection_tag =
      specfem::connections::type::nonconforming;
  constexpr static auto interface_tag = InterfaceTag;
  constexpr static auto boundary_tag = BoundaryTag;
  constexpr static auto wavefield = WavefieldType;
  const auto &nonconforming_interfaces = assembly.nonconforming_interfaces;
  const auto [self_edges, coupled_edges] =
      assembly.edge_types.get_edges_on_device(connection_tag, interface_tag,
                                              boundary_tag);

  if (self_edges.n_edges == 0 && coupled_edges.n_edges == 0)
    return;

  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  const auto num_points = assembly.mesh.element_grid.ngllx;

  using parallel_config = specfem::parallel_configuration::default_chunk_edge_config<
      DimensionTag, Kokkos::DefaultExecutionSpace>;

  // As written, field types cannot readily be defined in attributes. Define
  // them here.
  constexpr specfem::element::medium_tag self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();
  constexpr specfem::element::medium_tag coupled_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::coupled_medium();
  using CoupledFieldType = std::conditional_t<
      InterfaceTag == specfem::interface::interface_tag::acoustic_elastic,
      specfem::chunk_edge::displacement<parallel_config::chunk_size, NGLL,
                                        dimension_tag, coupled_medium,
                                        using_simd>,
      specfem::chunk_edge::acceleration<parallel_config::chunk_size, NGLL,
                                        dimension_tag, coupled_medium,
                                        using_simd> >;

  using CouplingTermsPack = specfem::chunk_edge::coupling_terms_pack<
      dimension_tag, interface_tag, boundary_tag, parallel_config::chunk_size,
      NGLL, NQuad_intersection>;
  using IntegrationFactor = specfem::chunk_edge::intersection_factor<
      dimension_tag, interface_tag, boundary_tag, parallel_config::chunk_size,
      NQuad_intersection>;

  using InterfaceFieldViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, parallel_config::chunk_size, NQuad_intersection,
      specfem::element::attributes<DimensionTag, self_medium>::components,
      using_simd, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_edges, coupled_edges);

  int scratch_size =
      CoupledFieldType::shmem_size() + CouplingTermsPack::shmem_size() +
      InterfaceFieldViewType::shmem_size() + IntegrationFactor::shmem_size();

  specfem::execution::for_each_level(
      "specfem::kokkos_kernels::impl::compute_coupling",
      chunk.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::index_type &chunk_iterator_index) {
        const auto &chunk_index = chunk_iterator_index.get_index();
        const auto &team = chunk_index.get_policy_index();
        const auto &self_chunk_iterator_index = chunk_index.get_self_index();
        const auto &coupled_chunk_iterator_index =
            chunk_index.get_coupled_index();
        const auto coupled_chunk_index =
            coupled_chunk_iterator_index.get_index();
        const auto self_chunk_index = self_chunk_iterator_index.get_index();

        CoupledFieldType coupled_field(team.team_scratch(0));
        specfem::assembly::load_on_device(coupled_chunk_index, field,
                                          coupled_field);

        CouplingTermsPack interface_data(team);

        specfem::assembly::load_on_device(
            self_chunk_index, nonconforming_interfaces, interface_data);
        InterfaceFieldViewType interface_field(team.team_scratch(0));

        team.team_barrier();
        specfem::medium::compute_coupling(self_chunk_index, interface_data,
                                          coupled_field, interface_field);

        IntegrationFactor integration_factor(team);

        specfem::assembly::load_on_device(
            self_chunk_index, nonconforming_interfaces, integration_factor);

        team.team_barrier();

        specfem::algorithms::coupling_integral(
            assembly, self_chunk_index, interface_field, integration_factor,
            [&](const auto &self_index, auto &self_field) {
              specfem::point::boundary<boundary_tag, dimension_tag, false>
                  point_boundary;
              specfem::assembly::load_on_device(self_index, assembly.boundaries,
                                                point_boundary);
              if constexpr (BoundaryTag == specfem::element::boundary_tag::
                                               acoustic_free_surface) {
                specfem::boundary_conditions::apply_boundary_conditions(
                    point_boundary, self_field);
              }

              specfem::assembly::atomic_add_on_device(self_index, field,
                                                      self_field);
            });
      });

  return;
}

template <specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          int NQuad_intersection,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_coupling(
    const specfem::assembly::assembly<DimensionTag> &assembly) {
  // Create dispatch tag for connection type
  using connection_dispatch =
      std::integral_constant<specfem::connections::type, ConnectionTag>;

  // Forward to implementation with dispatch tag
  if constexpr (ConnectionTag == specfem::connections::type::nonconforming) {
    compute_coupling<DimensionTag, WavefieldType, NGLL, NQuad_intersection,
                     InterfaceTag, BoundaryTag>(connection_dispatch(),
                                                assembly);
  } else {
    compute_coupling<DimensionTag, WavefieldType, InterfaceTag, BoundaryTag>(
        connection_dispatch(), assembly);
  }
}
