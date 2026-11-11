#include "conjugate_integral.hpp"
#include "specfem/algorithms/transfer_interpolate.hpp"
#include "specfem/element_coupling/accessor.hpp"
#include "specfem/element_coupling/attributes.hpp"

#include "specfem/boundary_conditions.hpp"
#include "specfem/chunk_face.hpp"
#include "specfem/datatype/chunk_face_view.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/element_coupling/to_string.hpp"
#include "specfem/execution.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/point.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <sstream>
#include <stdexcept>
#include <type_traits>

template <int NGLL, typename Tags>
void specfem::compute::impl::compute_coupling_conjugate_integral_nonconforming(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly)
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
{

  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto connection_tag =
      specfem::element_connections::type::nonconforming;
  constexpr static auto interface_tag = Tags::interface_tag;
  constexpr static auto boundary_tag = Tags::boundary_tag;
  constexpr static auto wavefield_tag = Tags::wavefield_tag;
  constexpr static auto flux_scheme_tag = Tags::flux_scheme_tag;

  // The pairs of this kernel's own container: self faces are the guest, the
  // coupled faces are the host. The self and the coupled face of one row carry
  // the same pair index, which addresses the row of the own container.
  const auto [self_intersections, coupled_intersections] =
      assembly.element_intersections.get_intersections_on_device(
          connection_tag, interface_tag, boundary_tag, flux_scheme_tag);

  if (self_intersections.N == 0 && coupled_intersections.N == 0) {
    return;
  }

  // Guest-face interpolants at the host nodes of the own pairs (the guest
  // container holds copies of the host container's rows, in the order of its
  // own list). If the container is empty, we have already returned above.
  const auto &own_container =
      assembly.nonconforming_interfaces
          .template get_interface_container<interface_tag, boundary_tag,
                                            connection_tag, flux_scheme_tag>();
  const auto interpolants = own_container.interpolants;

  if (own_container.self_is_host) {
    std::ostringstream oss;
    oss << "The guest-side nonconforming coupling kernel was called for a "
           "container whose self side is the host: interface "
        << specfem::element_coupling::to_string(interface_tag) << ", boundary "
        << specfem::element::to_string(boundary_tag);
    throw std::runtime_error(oss.str());
  }

  const auto field =
      assembly.fields.template get_simulation_field<wavefield_tag>();

  constexpr bool using_simd = false;
  using parallel_config =
      specfem::parallel_configuration::default_chunk_face_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>;
  constexpr int chunk_size = parallel_config::chunk_size;

  // As written, field types cannot readily be defined in attributes. Define
  // them here.
  constexpr specfem::element::medium_tag self_medium =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::self_medium();
  constexpr specfem::element::medium_tag coupled_medium =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::coupled_medium();
  constexpr int ncomp_self =
      specfem::element::attributes<dimension_tag, self_medium>::components;
  using CoupledFieldType = std::conditional_t<
      interface_tag ==
          specfem::element_coupling::interface_tag::acoustic_elastic,
      specfem::chunk_face::displacement<chunk_size, NGLL, dimension_tag,
                                        coupled_medium, using_simd>,
      specfem::chunk_face::acceleration<chunk_size, NGLL, dimension_tag,
                                        coupled_medium, using_simd>>;
  using CoupledPointTags =
      specfem::tags::Tags<dimension_tag, coupled_medium, using_simd>;
  using CoupledPointFieldType = std::conditional_t<
      interface_tag ==
          specfem::element_coupling::interface_tag::acoustic_elastic,
      specfem::point::displacement<CoupledPointTags>,
      specfem::point::acceleration<CoupledPointTags>>;
  using SelfPointFieldType = specfem::point::acceleration<
      specfem::tags::Tags<dimension_tag, self_medium, using_simd>>;
  // Flux at every host node of the chunk, (iface_local, ci, cj, icomp).
  using FluxViewType = specfem::datatype::VectorChunkFaceViewType<
      type_real, dimension_tag, chunk_size, NGLL, ncomp_self, using_simd,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  // Host-node data (face factor, normal, interpolants) of the own pairs: the
  // host GLL nodes are the quadrature points.
  using HostNodeCouplingData =
      specfem::element_coupling::accessor::coupling_terms_pack<
          dimension_tag, interface_tag, boundary_tag, flux_scheme_tag,
          chunk_size, NGLL, NGLL>;

  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_intersections, coupled_intersections);

  const int scratch_size =
      CoupledFieldType::shmem_size() + FluxViewType::shmem_size();

  specfem::execution::for_each_level(
      "specfem::compute::impl::compute_coupling_conjugate_integral",
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
        FluxViewType flux(team.team_scratch(0));
        specfem::assembly::load_on_device(coupled_chunk_index, field,
                                          coupled_field);

        const auto &nonconforming_interfaces =
            assembly.nonconforming_interfaces;
        const auto &boundaries = assembly.boundaries;

        team.team_barrier();

        // Flux into the guest medium, once per host node.
        specfem::execution::for_each_level(
            coupled_chunk_index.get_iterator(),
            [&](const typename std::decay_t<decltype(coupled_chunk_index)>::
                    iterator_type::index_type &iterator_index) {
              const auto &index = iterator_index.get_index();
              const auto &local_index = iterator_index.get_local_index();

              HostNodeCouplingData point_interface_data;
              specfem::assembly::load_on_device(index, nonconforming_interfaces,
                                                point_interface_data);

              CoupledPointFieldType coupled_point_field;
              for (int icomp = 0; icomp < CoupledPointFieldType::components;
                   icomp++) {
                coupled_point_field(icomp) =
                    coupled_field(local_index.iface, local_index.ipoint_i,
                                  local_index.ipoint_j, icomp);
              }

              SelfPointFieldType flux_point;
              specfem::medium_physics::compute_interface_flux<interface_tag>(
                  point_interface_data, coupled_point_field, flux_point);

              flux(local_index) = flux_point;
            });

        team.team_barrier();

        // Transposed interpolation of the flux onto each guest node.
        specfem::execution::for_each_level(
            self_chunk_index.get_iterator(),
            [&](const typename std::decay_t<decltype(self_chunk_index)>::
                    iterator_type::index_type &iterator_index) {
              const auto &index = iterator_index.get_index();
              const auto &local_index = iterator_index.get_local_index();

              SelfPointFieldType self_accel;
              specfem::algorithms::transfer_integrate(
                  index, local_index, interpolants, flux, self_accel);

              // The stored normal is the host's outward normal, opposite to
              // the guest's.
              for (int icomp = 0; icomp < ncomp_self; icomp++) {
                self_accel(icomp) = -self_accel(icomp);
              }

              // boundary_tag is the tag of the self (guest) element being
              // updated, so the free-surface condition applies only when self
              // is acoustic.
              if constexpr (boundary_tag == specfem::element::boundary_tag::
                                                acoustic_free_surface &&
                            self_medium ==
                                specfem::element::medium_tag::acoustic) {
                specfem::point::boundary<boundary_tag, dimension_tag, false>
                    point_boundary;
                specfem::assembly::load_on_device(index, boundaries,
                                                  point_boundary);
                specfem::boundary_conditions::apply_boundary_conditions(
                    point_boundary, self_accel);
              }
              specfem::assembly::atomic_add_on_device(index, field, self_accel);
            });
      });
}
