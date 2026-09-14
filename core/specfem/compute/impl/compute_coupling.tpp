#pragma once

#include "compute_coupling.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/algorithms/transfer_interpolate.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/boundary_conditions.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/chunk_face.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/data_access/accessor/point_accessor.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/enums.hpp"
#include "specfem/execution.hpp"
#include "specfem/execution/for_each_level.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/point.hpp"
#include "specfem/point/interface_index.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

#include "specfem/element_coupling/accessor.hpp"

namespace specfem::compute::impl {

template <int NGLL, int NQuad_intersection, typename Tags>
void compute_coupling_core_weakly_conforming(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {

  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto connection_tag =
      specfem::element_connections::type::weakly_conforming;
  constexpr static auto interface_tag = Tags::interface_tag;
  constexpr static auto boundary_tag = Tags::boundary_tag;
  constexpr static auto wavefield = Tags::wavefield_tag;
  constexpr static auto flux_scheme_tag = Tags::flux_scheme_tag;

  static_assert(flux_scheme_tag ==
                    specfem::element_coupling::flux_scheme_tag::natural,
                "Currently, we are enforcing only one flux scheme: natural");

  constexpr static auto self_medium =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::self_medium();

  const auto &conforming_interfaces = assembly.conforming_interfaces;
  const auto [self_intersections, coupled_intersections] =
      assembly.element_intersections.get_intersections_on_device(
          connection_tag, interface_tag, boundary_tag, flux_scheme_tag);

  if (self_intersections.N != coupled_intersections.N) {
    KOKKOS_ABORT_WITH_LOCATION(
        "Mismatch in number of self and coupled faces in compute_coupling.");
  }

  if (self_intersections.N == 0 && coupled_intersections.N == 0)
    return;

  const auto &field =
      assembly.fields.template get_simulation_field<wavefield>();
  const auto &boundaries = assembly.boundaries;

  const auto num_points = assembly.mesh.element_grid.ngllx;

  using parallel_config = std::conditional_t<
      dimension_tag == specfem::element::dimension_tag::dim2,
      specfem::parallel_configuration::default_chunk_edge_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>,
      specfem::parallel_configuration::default_chunk_face_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>>;

  using CoupledFieldType = typename specfem::element_coupling::attributes<
      dimension_tag, interface_tag>::template coupled_field_t<connection_tag>;
  using SelfFieldType = typename specfem::element_coupling::attributes<
      dimension_tag, interface_tag>::template self_field_t<connection_tag>;

  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension_tag, false>;

  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_intersections, coupled_intersections);

  specfem::execution::for_all(
      "specfem::compute::impl::compute_coupling", chunk,
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

        specfem::medium_physics::compute_coupling(point_interface_data,
                                                  coupled_field, self_field);

        PointBoundaryType point_boundary;
        specfem::assembly::load_on_device(index.self_index, boundaries,
                                          point_boundary);
        if constexpr (boundary_tag ==
                      specfem::element::boundary_tag::acoustic_free_surface) {
          specfem::boundary_conditions::apply_boundary_conditions(
              point_boundary, self_field);
        }

        specfem::assembly::atomic_add_on_device(index.self_index, field,
                                                self_field);
      });

  return;
}

template <int NGLL, int NQuad_intersection, typename Tags>
void compute_coupling_core_nonconforming(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {

  constexpr bool using_simd = false;

  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto connection_tag =
      specfem::element_connections::type::nonconforming;
  constexpr static auto interface_tag = Tags::interface_tag;
  constexpr static auto boundary_tag = Tags::boundary_tag;
  constexpr static auto wavefield = Tags::wavefield_tag;
  constexpr static auto flux_scheme_tag = Tags::flux_scheme_tag;

  static_assert(flux_scheme_tag ==
                    specfem::element_coupling::flux_scheme_tag::natural,
                "Currently, we are enforcing only one flux scheme: natural");

  const auto &nonconforming_interfaces = assembly.nonconforming_interfaces;
  const auto &boundaries = assembly.boundaries;

  const auto [self_intersections, coupled_intersections] =
      assembly.element_intersections.get_intersections_on_device(
          connection_tag, interface_tag, boundary_tag, flux_scheme_tag);

  if (self_intersections.N == 0 && coupled_intersections.N == 0)
    return;

  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  const auto num_points = assembly.mesh.element_grid.ngllx;

  using parallel_config = std::conditional_t<
      dimension_tag == specfem::element::dimension_tag::dim2,
      specfem::parallel_configuration::default_chunk_edge_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>,
      specfem::parallel_configuration::default_chunk_face_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>>;

  // As written, field types cannot readily be defined in attributes. Define
  // them here.
  constexpr specfem::element::medium_tag self_medium =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::self_medium();
  constexpr specfem::element::medium_tag coupled_medium =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::coupled_medium();
  using CoupledFieldType = std::conditional_t<
      dimension_tag == specfem::element::dimension_tag::dim2,
      std::conditional_t<
          interface_tag ==
              specfem::element_coupling::interface_tag::acoustic_elastic,
          specfem::chunk_edge::displacement<parallel_config::chunk_size, NGLL,
                                            dimension_tag, coupled_medium,
                                            using_simd>,
          specfem::chunk_edge::acceleration<parallel_config::chunk_size, NGLL,
                                            dimension_tag, coupled_medium,
                                            using_simd>>,
      std::conditional_t<
          interface_tag ==
              specfem::element_coupling::interface_tag::acoustic_elastic,
          specfem::chunk_face::displacement<parallel_config::chunk_size, NGLL,
                                            dimension_tag, coupled_medium,
                                            using_simd>,
          specfem::chunk_face::acceleration<parallel_config::chunk_size, NGLL,
                                            dimension_tag, coupled_medium,
                                            using_simd>>>;

  using CouplingTermsPack =
      specfem::element_coupling::accessor::coupling_terms_pack<
          dimension_tag, interface_tag, boundary_tag, flux_scheme_tag,
          parallel_config::chunk_size, NGLL, NQuad_intersection>;

  // should the nonconforming transfer be computed self-pointwise, or is it an
  // intersection-type, where the entire intersection needs to be computed
  // together?
  constexpr bool is_pointwise_coupling =
      specfem::data_access::is_point<CouplingTermsPack>::value;

  // teamwise integration data, only needed if not pointwise
  using IntegrationFactor = std::conditional_t<
      is_pointwise_coupling, specfem::data_access::EmptyAccessor,
      specfem::element_coupling::accessor::intersection_factor<
          dimension_tag, interface_tag, boundary_tag, flux_scheme_tag,
          parallel_config::chunk_size, NQuad_intersection>>;

  using InterfaceFieldViewType = std::conditional_t<
      is_pointwise_coupling, specfem::data_access::EmptyAccessor,
      specfem::datatype::VectorChunkEdgeViewType<
          type_real, dimension_tag, parallel_config::chunk_size,
          NQuad_intersection,
          specfem::element::attributes<dimension_tag, self_medium>::components,
          using_simd, Kokkos::DefaultExecutionSpace::scratch_memory_space,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;

  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_intersections, coupled_intersections);

  int scratch_size =
      CoupledFieldType::shmem_size() + CouplingTermsPack::shmem_size() +
      InterfaceFieldViewType::shmem_size() + IntegrationFactor::shmem_size();

  specfem::execution::for_each_level(
      "specfem::compute::impl::compute_coupling",
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

        const auto &nonconforming_interfaces =
            assembly.nonconforming_interfaces;
        const auto &assembly_mesh_xi = assembly.mesh.xi;
        const auto &boundaries = assembly.boundaries;

        // internal to lambda, since nvcc could not compile otherwise
        constexpr bool is_pointwise_coupling_ = is_pointwise_coupling;
        if constexpr (is_pointwise_coupling_) {
          team.team_barrier();
          specfem::execution::for_each_level(
              self_chunk_index.get_iterator(),
              [&](const typename std::decay_t<decltype(self_chunk_index)>::
                      iterator_type::index_type &iterator_index) {
                const auto &index = iterator_index.get_index();
                const auto &local_index = iterator_index.get_local_index();

                using SelfFieldType =
                    specfem::point::acceleration<specfem::tags::Tags<
                        dimension_tag, self_medium, false /*UseSIMD*/>>;

                CouplingTermsPack point_interface_data;
                specfem::assembly::load_on_device(
                    index, nonconforming_interfaces, point_interface_data);
                // TEMPORARY until we get rid of non-static interpolants
                point_interface_data.set_interpolants(
                    specfem::algorithms::LagrangeInterpolant(assembly_mesh_xi));

                SelfFieldType self_field;
                specfem::medium_physics::compute_coupling(
                    local_index, point_interface_data, coupled_field,
                    self_field);

                specfem::point::boundary<boundary_tag, dimension_tag, false>
                    point_boundary;
                specfem::assembly::load_on_device(index, boundaries,
                                                  point_boundary);
                if constexpr (boundary_tag == specfem::element::boundary_tag::
                                                  acoustic_free_surface) {
                  specfem::boundary_conditions::apply_boundary_conditions(
                      point_boundary, self_field);
                }
                specfem::assembly::atomic_add_on_device(index, field,
                                                        self_field);
              });

        } else {

          CouplingTermsPack interface_data(team);

          specfem::assembly::load_on_device(
              self_chunk_index, nonconforming_interfaces, interface_data);
          InterfaceFieldViewType interface_field(team.team_scratch(0));

          team.team_barrier();
          specfem::medium_physics::compute_coupling(
              self_chunk_index, interface_data, coupled_field, interface_field);

          IntegrationFactor integration_factor(team);

          specfem::assembly::load_on_device(
              self_chunk_index, nonconforming_interfaces, integration_factor);

          team.team_barrier();

          specfem::algorithms::coupling_integral(
              assembly.nonconforming_interfaces, self_chunk_index,
              interface_field, integration_factor,
              [&](const auto &self_index, auto &self_field) {
                specfem::point::boundary<boundary_tag, dimension_tag, false>
                    point_boundary;
                specfem::assembly::load_on_device(
                    self_index, assembly.boundaries, point_boundary);
                if constexpr (boundary_tag == specfem::element::boundary_tag::
                                                  acoustic_free_surface) {
                  specfem::boundary_conditions::apply_boundary_conditions(
                      point_boundary, self_field);
                }

                specfem::assembly::atomic_add_on_device(self_index, field,
                                                        self_field);
              });
        }
      });

  return;
}

template <int NGLL, int NQuad_intersection, typename Tags>
void compute_coupling_core(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {
  constexpr auto connection_tag = Tags::connection_tag;
  if constexpr (connection_tag ==
                specfem::element_connections::type::nonconforming) {
    compute_coupling_core_nonconforming<NGLL, NQuad_intersection, Tags>(
        assembly);
  } else {
    compute_coupling_core_weakly_conforming<NGLL, NQuad_intersection, Tags>(
        assembly);
  }
}

template <int NGLL, typename Tags>
void compute_coupling(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {

  constexpr auto WavefieldType = Tags::wavefield_tag;

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          CONNECTION_SET(weakly_conforming, nonconforming) *
          INTERFACE_SET(elastic_acoustic, acoustic_elastic) *
          BOUNDARY_SET(none, acoustic_free_surface, stacey,
                       composite_stacey_dirichlet) *
          FLUX_SCHEME_SET(natural),
      [&]<typename ElementTags>() {
        constexpr auto self_medium = specfem::element_coupling::attributes<
            ElementTags::dimension_tag,
            ElementTags::interface_tag>::self_medium();
        if constexpr (self_medium == Tags::medium_tag) {
          compute_coupling_core<
              NGLL, NGLL,
              specfem::tags::Tags<
                  ElementTags::dimension_tag, ElementTags::connection_tag,
                  WavefieldType, ElementTags::interface_tag,
                  ElementTags::boundary_tag, ElementTags::flux_scheme_tag>>(
              assembly);
        }
      });
}

} // namespace specfem::compute::impl
