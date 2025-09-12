#pragma once

#include "compute_coupling.hpp"
#include "enumerations/interface.hpp"
#include "execution/chunked_intersection_iterator.hpp"
#include "execution/for_all.hpp"
#include "medium/compute_coupling.hpp"
#include "parallel_configuration/chunk_edge_config.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

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
  const auto &coupled_interfaces = assembly.coupled_interfaces2;
  const auto [self_edges, coupled_edges] =
      assembly.edge_types.get_edges_on_device(connection_tag, interface_tag,
                                              boundary_tag);

  if (self_edges.extent(0) == 0 && coupled_edges.extent(0) == 0)
    return;

  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  const auto num_points = assembly.mesh.ngllx;

  using parallel_config = specfem::parallel_config::default_chunk_edge_config<
      DimensionTag, Kokkos::DefaultExecutionSpace>;

  using CoupledFieldType = typename specfem::interface::attributes<
      dimension_tag, interface_tag>::template coupled_field_t<connection_tag>;
  using SelfFieldType = typename specfem::interface::attributes<
      dimension_tag, interface_tag>::template self_field_t<connection_tag>;

  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_edges, coupled_edges, num_points);

  specfem::execution::for_all(
      "specfem::kokkos_kernels::impl::compute_coupling", chunk,
      KOKKOS_LAMBDA(const typename decltype(chunk)::base_index_type &index) {
        const auto self_index = index.self_index;
        const auto coupled_index = index.coupled_index;

        specfem::point::coupled_interface<dimension_tag, connection_tag,
                                          interface_tag, boundary_tag>
            point_interface_data;
        specfem::assembly::load_on_device(self_index, coupled_interfaces,
                                          point_interface_data);

        CoupledFieldType coupled_field;
        specfem::assembly::load_on_device(coupled_index, field, coupled_field);

        SelfFieldType self_field;

        specfem::medium::compute_coupling(point_interface_data, coupled_field,
                                          self_field);

        specfem::assembly::atomic_add_on_device(self_index, field, self_field);
      });

  return;
}

// template <specfem::dimension::type DimensionTag,
//           specfem::wavefield::simulation_field WavefieldType,
//           specfem::interface::interface_tag InterfaceTag,
//           specfem::element::boundary_tag BoundaryTag>
// void specfem::kokkos_kernels::impl::compute_coupling(
//     std::integral_constant<
//         specfem::connections::type,
//         specfem::connections::type::nonconforming> /*unused*/,
//     const specfem::assembly::assembly<DimensionTag> &assembly) {

//   constexpr static auto dimension_tag = DimensionTag;
//   constexpr static auto connection_tag =
//       specfem::connections::type::nonconforming;
//   constexpr static auto interface_tag = InterfaceTag;
//   constexpr static auto boundary_tag = BoundaryTag;
//   constexpr static auto wavefield = WavefieldType;
//   const auto &coupled_interfaces = assembly.coupled_interfaces2;
//   const auto [self_edges, coupled_edges] =
//       assembly.edge_types.get_edges_on_device(connection_tag, interface_tag,
//                                               boundary_tag);

//   if (self_edges.extent(0) == 0 && coupled_edges.extent(0) == 0)
//     return;

//   const auto field = assembly.fields.template get_simulation_field<wavefield>();

//   const auto num_points = assembly.mesh.ngllx;

//   using parallel_config = specfem::parallel_config::default_chunk_edge_config<
//       DimensionTag, Kokkos::DefaultExecutionSpace>;

//   using CoupledFieldType = typename specfem::interface::attributes<
//       dimension_tag, interface_tag>::template coupled_field_t<connection_tag>;
//   using SelfFieldType = typename specfem::interface::attributes<
//       dimension_tag, interface_tag>::template self_field_t<connection_tag>;

//   specfem::execution::ChunkedIntersectionIterator chunk(
//       parallel_config(), self_edges, coupled_edges, num_points);

//   specfem::execution::for_each_level(
//       "specfem::kokkos_kernels::impl::compute_coupling", chunk,
//       KOKKOS_LAMBDA(
//           const typename decltype(chunk)::index_type &chunk_iterator_index) {
//         const auto &chunk_index = chunk_iterator_index.get_index();
//         const auto &self_chunk_index = chunk_index.self_index;
//         const auto &coupled_chunk_index = chunk_index.coupled_index;

//         CoupledFieldType coupled_field; //< Coupled field type should be chunk
//                                         // edge field type
//         specfem::assembly::load_on_device(coupled_chunk_index, field,
//                                           coupled_field);

//         specfem::chunk_element::coupled_interface<dimension_tag, connection_tag,
//                                                   interface_tag, boundary_tag>
//             interface_data;
//         specfem::assembly::load_on_device(self_chunk_index, coupled_interfaces,
//                                           interface_data);

//         specfem::execution::for_each_level(
//             chunk_index.get_iterator(),
//             [&](const typename decltype(
//                 chunk_index)::iterator_type::index_type &iterator_index) {
//               const auto index = iterator_index.get_index();
//               const auto self_index = index.self_index;

//               // @ Kentaro compute the coupling using edge data and transfer
//               // functions here

//               SelfFieldType self_field;
//               specfem::medium::compute_coupling(index, point_interface_data,
//                                                 coupled_field, self_field);

//               if (BoundaryTag ==
//                   specfem::element::boundary_tag::acoustic_free_surface) {
//                 specfem::point::boundary<dimension_tag, boundary_tag, false>
//                     point_boundary;
//                 specfem::assembly::load_on_device(
//                     self_index, assembly.boundaries, point_boundary);
//                 specfem::medium::apply_boundary_conditions(point_boundary,
//                                                            self_field);
//               }

//               specfem::assembly::atomic_add_on_device(self_index, field,
//                                                       self_field);
//             });
//       });

//   return;
// }
