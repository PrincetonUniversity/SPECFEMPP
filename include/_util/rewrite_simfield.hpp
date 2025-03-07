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

#include "mesh/coupled_interfaces/interface_container.hpp"
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag Medium1,
          specfem::element::medium_tag Medium2>
void clear_couplings(specfem::mesh::interface_container<DimensionType, Medium1,
                                                        Medium2> &container,
                     specfem::compute::mesh_to_compute_mapping &mapping,
                     specfem::adjacency_graph::adjacency_graph<4> &graph) {
  int placement = 0;
  for (int i = 0; i < container.num_interfaces; i++) {
    int aspec = container.medium1_index_mapping(i);
    int bspec = container.medium2_index_mapping(i);
    int aspec_compute = mapping.mesh_to_compute(aspec);
    int bspec_compute = mapping.mesh_to_compute(bspec);
    // int elem0 = graph.get_adjacency(aspec_compute,0).elem;
    // int elem1 = graph.get_adjacency(aspec_compute,1).elem;
    // int elem2 = graph.get_adjacency(aspec_compute,2).elem;
    // int elem3 = graph.get_adjacency(aspec_compute,3).elem;
    // check if they are adjacent
    if (graph.get_adjacency(aspec_compute, 0).elem == bspec_compute ||
        graph.get_adjacency(aspec_compute, 1).elem == bspec_compute ||
        graph.get_adjacency(aspec_compute, 2).elem == bspec_compute ||
        graph.get_adjacency(aspec_compute, 3).elem == bspec_compute) {
      // they are, so don't copy
      //  removals ++;
    } else {
      // not, so copy over
      container.medium1_index_mapping(placement) = aspec;
      container.medium2_index_mapping(placement) = bspec;
      placement++;
    }
  }
  container.num_interfaces = placement;
  Kokkos::resize(container.medium1_index_mapping, placement);
  Kokkos::resize(container.medium2_index_mapping, placement);
}

void remap_with_disconts(
    specfem::compute::assembly &assembly,
    _util::demo_assembly::simulation_params &params,
    std::vector<specfem::adjacency_graph::adjacency_pointer> &removals,
    bool disconnect_different_media = true) {

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

  auto mesh = params.get_mesh();
  const auto are_same_mats = [&](int a_ispec, int b_ispec) -> bool {
    const int a_ispec_mesh = assembly.mesh.mapping.compute_to_mesh(a_ispec);
    const int b_ispec_mesh = assembly.mesh.mapping.compute_to_mesh(b_ispec);
    const auto &a_matspec = mesh.materials.material_index_mapping(a_ispec_mesh);
    const auto &b_matspec = mesh.materials.material_index_mapping(b_ispec_mesh);
    return a_matspec.type == b_matspec.type &&
           a_matspec.property == b_matspec.property &&
           a_matspec.index == b_matspec.index;
  };
  if (disconnect_different_media) {
    for (int ispec = 0; ispec < assembly.mesh.nspec; ++ispec) {
      for (int iedge = 0; iedge < 4; iedge++) {
        specfem::adjacency_graph::adjacency_pointer adj =
            graph.get_adjacency(ispec, iedge);
        if (adj.is_active() && !are_same_mats(ispec, adj.elem)) {
          removals.push_back(adj);
        }
      }
    }
  }

  // remove edges
  clear_couplings(mesh.coupled_interfaces.elastic_acoustic,
                  assembly.mesh.mapping, graph);
  clear_couplings(mesh.coupled_interfaces.elastic_poroelastic,
                  assembly.mesh.mapping, graph);
  clear_couplings(mesh.coupled_interfaces.acoustic_poroelastic,
                  assembly.mesh.mapping, graph);
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
          specfem::wavefield::simulation_field WavefieldType>
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
