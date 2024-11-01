#ifndef __UTIL_REWRITE_SIMFIELD_HPP_
#define __UTIL_REWRITE_SIMFIELD_HPP_

#include "adjacency_graph/adjacency_graph.hpp"
#include "compute/fields/simulation_field.hpp"

#include "compute/assembly/assembly.hpp"
#include "mesh/mesh.hpp"
#include "policies/chunk.hpp"

#include "dump_simfield.hpp"

#include <Kokkos_Core.hpp>
#include <functional>

void remap_with_disconts(
    specfem::compute::assembly &assembly,
    _util::demo_assembly::simulation_params &params,
    const std::vector<specfem::adjacency_graph::adjacency_pointer> &removals) {

  // specfem::compute::simulation_field<WavefieldType> & field =
  // assembly.fields;
  // remap_with_disconts(assembly.fields.forward,assembly.mesh,assembly.properties,edge_removals);
  using ViewType =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  auto graph = specfem::adjacency_graph::from_index_mapping(
      assembly.mesh.points.h_index_mapping);
  specfem::adjacency_graph::adjacency_pointer null_adj;

#ifdef _EVENT_MARCHER_DUMPS_
  _util::dump_adjacency_graph(_index_change_dump_ + "/prior_graph.dat", graph);
#endif

  // remove edges
  for (int i = 0; i < removals.size(); i++) {
    graph.form_adjacency(removals[i], null_adj);
  }

#ifdef _EVENT_MARCHER_DUMPS_
  _util::dump_adjacency_graph(_index_change_dump_ + "/post_graph.dat", graph);
#endif

  // rebuild index_mapping
  int nglob;
  ViewType index_mapping = specfem::adjacency_graph::to_index_mapping(
      graph, assembly.mesh.ngllz, assembly.mesh.ngllx, &nglob);
  ViewType::HostMirror h_index_mapping =
      ViewType::HostMirror(Kokkos::create_mirror_view(index_mapping));

  // push everyting back (as in compute/compute_assembly.cpp)
  assembly.mesh.points.index_mapping = index_mapping;
  assembly.mesh.points.h_index_mapping = h_index_mapping;

  auto mesh = params.get_mesh();
  auto sources = params.get_sources();
  auto receivers = params.get_receivers();
  auto stypes = params.get_seismogram_types();
  auto t0 = params.get_t0();
  auto dt = params.get_dt();
  auto max_timesteps = params.get_numsteps();
  auto max_sig_step = params.get_num_seismogram_steps();
  auto simulation = params.get_simulation_type();

  assembly.partial_derivatives = { assembly.mesh };
  assembly.properties = { assembly.mesh.nspec, assembly.mesh.ngllz,
                          assembly.mesh.ngllx, assembly.mesh.mapping,
                          mesh.tags,           mesh.materials };
  assembly.kernels = { assembly.mesh.nspec, assembly.mesh.ngllz,
                       assembly.mesh.ngllx, assembly.mesh.mapping, mesh.tags };
  assembly.sources = { sources,
                       assembly.mesh,
                       assembly.partial_derivatives,
                       assembly.properties,
                       t0,
                       dt,
                       max_timesteps };
  assembly.receivers = { max_sig_step, receivers, stypes, assembly.mesh };
  assembly.boundaries = { assembly.mesh.nspec,   assembly.mesh.ngllz,
                          assembly.mesh.ngllx,   mesh,
                          assembly.mesh.mapping, assembly.mesh.quadratures,
                          assembly.properties,   assembly.partial_derivatives };
  assembly.coupled_interfaces = { mesh,
                                  assembly.mesh.points,
                                  assembly.mesh.quadratures,
                                  assembly.partial_derivatives,
                                  assembly.properties,
                                  assembly.mesh.mapping };
  assembly.fields = { assembly.mesh, assembly.properties, simulation };
  assembly.boundary_values = { max_timesteps, assembly.mesh,
                               assembly.properties, assembly.boundaries };
}

// TODO change return type to allow for 2+ component media
template <specfem::element::medium_tag MediumTag,
          specfem::wavefield::type WavefieldType>
void set_field_disp(
    specfem::compute::simulation_field<WavefieldType> &field,
    const specfem::compute::mesh &mesh,
    std::function<type_real(const type_real, const type_real)> dispfunc) {

  // https://specfem2d-kokkos.readthedocs.io/en/devel/api/policies/chunk_element/index.html#
  const int nspec = mesh.nspec;
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> elements("elements",
                                                              nspec);
  Kokkos::parallel_for(
      "generate_elements", nspec,
      KOKKOS_LAMBDA(const int ispec) { elements(ispec) = ispec; });
  constexpr auto dim2 = specfem::dimension::type::dim2;
  using SIMD = specfem::datatype::simd<type_real, false>;
  using ParallelConfiguration = specfem::parallel_config::default_chunk_config<
      dim2, SIMD, Kokkos::DefaultExecutionSpace>;
  using CoordinateAccess = specfem::point::global_coordinates<dim2>;
  using PointDisplacementType =
      specfem::point::field<dim2, MediumTag, true, false, false, false,
                            SIMD::using_simd>;
  using ChunkPolicyType = specfem::policy::element_chunk<ParallelConfiguration>;

  constexpr int simd_size = SIMD::size();

  ChunkPolicyType chunk_policy(elements, mesh.ngllz, mesh.ngllx);

  const auto &policy =
      static_cast<typename ChunkPolicyType::policy_type &>(chunk_policy);
  Kokkos::parallel_for(
      "set_field_disp", policy,
      KOKKOS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
        for (int tile = 0; tile < ParallelConfiguration::tile_size;
             tile += ParallelConfiguration::chunk_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicyType::tile_size * simd_size +
              tile;
          const auto iterator =
              chunk_policy.league_iterator(starting_element_index);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [&](const int i) {
                const auto iterator_index = iterator(i);
                const auto index = iterator_index.index;

                // can't find a load_on_device for this
                CoordinateAccess coordinates(
                    mesh.points.coord(0, index.ispec, index.iz, index.ix),
                    mesh.points.coord(1, index.ispec, index.iz, index.ix));
                // specfem::compute::load_on_device(index, coordinates,
                // mesh.points);
                PointDisplacementType disp;
                disp.displacement(0) = dispfunc(coordinates.x, coordinates.z);

                // std::cout << "setting f(" << coordinates.x << "," <<
                // coordinates.z << ") = " << disp.displacement(0) << "\n";
                specfem::compute::store_on_device(index, disp, field);
              });
        }
      });
}

#endif
