#pragma once

#include "specfem/assembly.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/execution.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::algorithms {

/**
 * @brief Takes a field `intersection_field` on the intersection and computes,
 * for each self GLL point, the integral of `intersection_field` times the
 * normal derivative of the shape function at that point. `intersection_field`
 * should be call-accessible (e.g. Kokkos::View) with shape:
 *
 * (chunk_size, n_quad_intersection, self::components())
 *
 * After handling any other intersection forces, boundary conditions, etc. the
 * result can be `atomic_add`ed to the acceleration field.
 *
 * @tparam dimension_tag dimension of the simulation
 * @param nonconforming_interfaces - assembly.nonconforming_interfaces struct
 */
template <specfem::element_coupling::interface_tag interface_tag,
          specfem::element::boundary_tag boundary_tag,
          specfem::element::dimension_tag dimension_tag>
Kokkos::View<type_real ****, Kokkos::DefaultExecutionSpace>
shape_function_self_normal_derivatives(
    const specfem::assembly::edge_types<dimension_tag> &edge_types,
    const specfem::assembly::mesh<dimension_tag> &mesh,
    const specfem::assembly::nonconforming_interfaces<dimension_tag>
        &nonconforming_interfaces) {
  using ReturnViewType =
      Kokkos::View<type_real ****, Kokkos::DefaultExecutionSpace>;

  const auto [self_edges, coupled_edges] = edge_types.get_edges_on_device(
      specfem::element_connections::type::nonconforming, interface_tag,
      boundary_tag);
  const auto &element_grid = mesh.element_grid;

  // TODO: get nquad_intersection from somewhere else
  const int ngllz = element_grid.ngllz;
  const int ngllx = element_grid.ngllx;
  const int nquad_intersection = std::max(ngllz, ngllx);
  ReturnViewType normal_derivs("shape_function_self_normal_derivatives",
                               self_edges.n_edges, ngllz, ngllx,
                               nquad_intersection);

  //   using ParallelConfig =
  //   specfem::parallel_configuration::default_chunk_config<
  //       dimension_tag, specfem::datatype::simd<type_real, false
  //       /*using_simd*/>, Kokkos::DefaultExecutionSpace>;
  //   specfem::execution::ChunkedDomainIterator chunk(
  //       ParallelConfig(), self_edges.element_index, element_grid);

  using parallel_config =
      specfem::parallel_configuration::default_chunk_edge_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>;
  specfem::execution::ChunkedEdgeIterator chunk(parallel_config(), self_edges);

  return normal_derivs; // remove when done

  specfem::execution::for_each_level(
      "specfem::compute::compute_stiffness_interaction", chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::index_type &chunk_iterator_index) {
        const auto &chunk_index = chunk_iterator_index.get_index();
        const auto &team = chunk_index.get_policy_index();
        const int &num_edges = chunk_index.nedges();

        specfem::execution::for_each_level(
            specfem::execution::TeamThreadMDRangeIterator(team, num_edges,
                                                          ngllz, ngllx),
            [&](const auto &index) {
              const int iedge = index(0);
              const int iz = index(1);
              const int ix = index(2);

              // this index will always have ipoint = 0, but we will not use it
              const auto edge_index =
                  chunk_index.get_iterator()(iedge).get_index();

              const bool ipoint_n_is_ix =
                  edge_index.edge_type ==
                      specfem::mesh_entity::dim2::type::left ||
                  edge_index.edge_type ==
                      specfem::mesh_entity::dim2::type::right;
              const int ipoint_s = ipoint_n_is_ix ? iz : ix;
              const int ipoint_n = ipoint_n_is_ix ? ix : iz;

              // TODO find access to lagrange_derivative and uncomment
              const type_real dlagrange_dn = 0;
              //       = -(ipoint_n_is_ix ? lagrange_derivative.xi(0, ipoint_n)
              //                        : lagrange_derivative.gamma(0,
              //                        ipoint_n));

              for (int iquad = 0; iquad < nquad_intersection; iquad++) {

                // dshape_dn, local-normal derivative (derivative of
                // normal-direction L, which is normally kronecker delta
                // indicating on edge)

                // first, in the local-normal component
                type_real dshape_dn = 0;
                // = intersection_normal_contravariant_edgelocal(iedge, iquad,
                // 0) * (dlagrange_dn * transfer_function_self(iquad));
                // dshape_dn, local-tangential derivative (differentiate
                // transfer_function_self instead of normal-direction L)
                if (ipoint_n == 0) {
                  // the local-tangential is constant zero if we are not on the
                  // edge.
                  //   dshape_dn +=
                  //       intersection_normal_contravariant_edgelocal(iedge,
                  //       iquad, 1) * transfer_function_self_derivative(iedge,
                  //       ipoint_s, iquad);
                }

                normal_derivs(edge_index.iedge, iz, ix, iquad) = dshape_dn;
              }
            });
        // specfem::execution::for_each_level(
        //     chunk_index.get_iterator(),
        //     [&](const typename ChunkIndexType::iterator_type::index_type
        //             &iterator_index) {
        //       const auto index = iterator_index.get_index();
        //       const auto index_local = iterator_index.get_local_index();

        //       index.ielem
        //     });
      });

  return normal_derivs;
}

} // namespace specfem::algorithms
