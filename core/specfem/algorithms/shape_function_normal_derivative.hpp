#pragma once

#include "specfem/assembly.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/chunk_edge/nonconforming_interface.hpp"
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
          specfem::element::boundary_tag boundary_tag, int nquad_element_,
          int nquad_intersection_, typename SelfEdgeListView,
          typename HPrimeView, typename TransformedIntersectionNormalView>
Kokkos::View<type_real ****, Kokkos::DefaultExecutionSpace>
shape_function_self_normal_derivatives(
    const SelfEdgeListView &self_edges,
    const specfem::assembly::nonconforming_interfaces<
        specfem::element::dimension_tag::dim2> &nonconforming_interfaces,
    const int &ngllz, const int &ngllx, const HPrimeView &hprime,
    const TransformedIntersectionNormalView
        &intersection_normal_contravariant_edgelocal) {
  const auto dimension_tag = specfem::element::dimension_tag::dim2;
  using ReturnViewType =
      Kokkos::View<type_real ****, Kokkos::DefaultExecutionSpace>;

  // TODO: get nquad_intersection from somewhere else
  const int nquad_intersection = nquad_intersection_;
  if (nquad_intersection != nquad_intersection_) {
    throw std::runtime_error(
        std::string("shape_function_self_normal_derivatives() kernel run with "
                    "nquad_intersection = ") +
        std::to_string(nquad_intersection_) +
        ", but assembly got nquad_intersection == " +
        std::to_string(nquad_intersection));
  }
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

  using TransferFunctionSelf = specfem::chunk_edge::transfer_function_self<
      dimension_tag, interface_tag, boundary_tag, parallel_config::chunk_size,
      nquad_intersection_, nquad_element_>;

  specfem::execution::for_each_level(
      "specfem::compute::shape_function_normal_derivative",
      chunk.set_scratch_size(
          0, Kokkos::PerTeam(TransferFunctionSelf::shmem_size())),
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::index_type &chunk_iterator_index) {
        const auto &chunk_index = chunk_iterator_index.get_index();
        const auto &team = chunk_index.get_policy_index();
        const int &num_edges = chunk_index.nedges();

        TransferFunctionSelf transfer_function_self(team);
        specfem::assembly::load_on_device(chunk_index, nonconforming_interfaces,
                                          transfer_function_self);

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
              int ipoint_n = ipoint_n_is_ix ? ix : iz;
              const int ngll_n = ipoint_n_is_ix ? ngllx : ngllz;
              const int ngll_s = ipoint_n_is_ix ? ngllz : ngllx;

              if (edge_index.edge_type ==
                      specfem::mesh_entity::dim2::type::right ||
                  edge_index.edge_type ==
                      specfem::mesh_entity::dim2::type::top) {
                // we want 0 to be on the edge
                ipoint_n = ngll_n - 1 - ipoint_n;
              }

              const type_real dlagrange_dn = -hprime(0, ipoint_n);

              for (int iquad = 0; iquad < nquad_intersection; iquad++) {

                // dshape_dn, local-normal derivative (derivative of
                // normal-direction L, which used to be kronecker delta
                // indicating on edge. Now has dlagrange_dn != 0 for all
                // ipoint_n)

                // first, in the local-normal component
                type_real dshape_dn =
                    intersection_normal_contravariant_edgelocal(
                        edge_index.iedge, iquad, 0) *
                    (dlagrange_dn *
                     transfer_function_self(iedge, ipoint_s, iquad));
                // dshape_dn, local-tangential derivative (differentiate
                // transfer_function_self instead of normal-direction L)
                if (ipoint_n == 0) {
                  // the local-tangential is constant zero if we are not on the
                  // edge.

                  // recover L'  (at point ipoint_s) on intersection quadrature
                  // points: use transfer function.
                  type_real transfer_function_derivative = 0;
                  for (int ipoint_s2 = 0; ipoint_s2 < ngll_s; ipoint_s2++) {
                    transfer_function_derivative +=
                        hprime(ipoint_s2, ipoint_s) *
                        transfer_function_self(iedge, ipoint_s2, iquad);
                  }

                  dshape_dn += intersection_normal_contravariant_edgelocal(
                                   edge_index.iedge, iquad, 1) *
                               transfer_function_derivative;
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
