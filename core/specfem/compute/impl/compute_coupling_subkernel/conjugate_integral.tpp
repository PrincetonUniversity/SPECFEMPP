#include "conjugate_integral.hpp"
#include "specfem/element_coupling/accessor.hpp"
#include "specfem/element_coupling/attributes.hpp"

#include "specfem/boundary_conditions.hpp"
#include "specfem/chunk_face.hpp"
#include "specfem/datatype/accessor_type.hpp"
#include "specfem/datatype/chunk_face_view.hpp"
#include "specfem/datatype/register_array.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/medium_physics.hpp"

// empty for dim2 for now
template <int NGLL, typename Tags>
void specfem::compute::impl::compute_coupling_conjugate_integral_nonconforming(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly)
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim2)
{}

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

  constexpr static auto conjugate_interface_tag =
      specfem::element_coupling::attributes<
          dimension_tag, interface_tag>::conjugate_interface();

  const auto [conjugate_coupled_intersections, conjugate_self_intersections] =
      assembly.element_intersections.get_intersections_on_device(
          connection_tag, conjugate_interface_tag, boundary_tag,
          flux_scheme_tag);

  if (conjugate_self_intersections.N == 0 &&
      conjugate_coupled_intersections.N == 0) {
    return;
  }

  const auto field =
      assembly.fields.template get_simulation_field<wavefield_tag>();

  using parallel_config = std::conditional_t<
      dimension_tag == specfem::element::dimension_tag::dim2,
      specfem::parallel_configuration::default_chunk_edge_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>,
      specfem::parallel_configuration::default_chunk_face_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>>;
  constexpr bool using_simd = false;
  constexpr int NQuad_intersection = NGLL;

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
      specfem::chunk_face::displacement<parallel_config::chunk_size, NGLL,
                                        dimension_tag, coupled_medium,
                                        using_simd>,
      specfem::chunk_face::acceleration<parallel_config::chunk_size, NGLL,
                                        dimension_tag, coupled_medium,
                                        using_simd>>;
  using CoupledPointFieldType = std::conditional_t<
      interface_tag ==
          specfem::element_coupling::interface_tag::acoustic_elastic,
      specfem::point::displacement<
          specfem::tags::Tags<dimension_tag, coupled_medium, using_simd>>,
      specfem::point::acceleration<
          specfem::tags::Tags<dimension_tag, coupled_medium, using_simd>>>;

  using ConjugateCouplingData =
      specfem::element_coupling::accessor::coupling_terms_pack<
          dimension_tag, conjugate_interface_tag, boundary_tag, flux_scheme_tag,
          parallel_config::chunk_size, NGLL, NQuad_intersection>;

  // should the nonconforming transfer be computed self-pointwise, or is it an
  // intersection-type, where the entire intersection needs to be computed
  // together?
  constexpr bool is_pointwise_coupling =
      specfem::data_access::is_point<ConjugateCouplingData>::value;
  static_assert(is_pointwise_coupling == true);

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
      parallel_config(), conjugate_self_intersections,
      conjugate_coupled_intersections);

  int scratch_size = CoupledFieldType::shmem_size();

  specfem::execution::for_each_level(
      "compute_coupling_extra_kernel_symmetrizer",
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

        team.team_barrier();
        specfem::execution::for_each_level(
            self_chunk_index.get_iterator(),
            [&](const typename std::decay_t<decltype(self_chunk_index)>::
                    iterator_type::index_type &iterator_index) {
              const auto &index = iterator_index.get_index();
              const auto &iface_global = index.iface;
              const auto &local_index = iterator_index.get_local_index();
              const auto &iface_local = local_index.iface;
              using SelfPointFieldType =
                  specfem::point::acceleration<specfem::tags::Tags<
                      dimension_tag, self_medium, false /*UseSIMD*/>>;

              SelfPointFieldType self_accel;
              for (int icomp = 0; icomp < ncomp_self; icomp++) {
                self_accel(icomp) = 0;
              }

              // =====================
              // interpolate this shape function onto the coupled

              for (int ipoint_coupled = 0; ipoint_coupled < NGLL;
                   ipoint_coupled++) {
                for (int jpoint_coupled = 0; jpoint_coupled < NGLL;
                     jpoint_coupled++) {
                  const auto coupled_index = conjugate_coupled_intersections(
                      iface_global)(ipoint_coupled, jpoint_coupled);

                  ConjugateCouplingData point_interface_data;
                  specfem::assembly::load_on_device(coupled_index,
                                                    nonconforming_interfaces,
                                                    point_interface_data);
                  // TEMPORARY until we get rid of non-static interpolants
                  point_interface_data.set_interpolants(
                      specfem::algorithms::LagrangeInterpolant(
                          assembly_mesh_xi));

                  CoupledPointFieldType coupled_point_field;
                  specfem::assembly::load_on_device(coupled_index, field,
                                                    coupled_point_field);

                  // transfer self shape function to coupled side.
                  const type_real interpolated_shape_function =
                      point_interface_data.interpolants(index.ipoint_i, 0) *
                      point_interface_data.interpolants(index.ipoint_j, 1);

                  // accumulate self_accel by medium_physics::compute_coupling
                  if constexpr (interface_tag ==
                                specfem::element_coupling::interface_tag::
                                    acoustic_elastic) {
                    // negative sign, since the normal is in the opposite
                    // direction.
                    self_accel(0) -=
                        interpolated_shape_function *
                        point_interface_data.face_factor *
                        (point_interface_data.face_normal(0) *
                             coupled_field(iface_local, ipoint_coupled,
                                           jpoint_coupled, 0) +
                         point_interface_data.face_normal(1) *
                             coupled_field(iface_local, ipoint_coupled,
                                           jpoint_coupled, 1) +
                         point_interface_data.face_normal(2) *
                             coupled_field(iface_local, ipoint_coupled,
                                           jpoint_coupled, 2));
                  } else {
                    // negative sign, since the normal is in the opposite
                    // direction.
                    self_accel(0) -= interpolated_shape_function *
                                     point_interface_data.face_factor *
                                     (point_interface_data.face_normal(0) *
                                      coupled_field(iface_local, ipoint_coupled,
                                                    jpoint_coupled, 0));
                    self_accel(1) -= interpolated_shape_function *
                                     point_interface_data.face_factor *
                                     (point_interface_data.face_normal(1) *
                                      coupled_field(iface_local, ipoint_coupled,
                                                    jpoint_coupled, 0));
                    self_accel(2) -= interpolated_shape_function *
                                     point_interface_data.face_factor *
                                     (point_interface_data.face_normal(2) *
                                      coupled_field(iface_local, ipoint_coupled,
                                                    jpoint_coupled, 0));
                  }
                }
              }

              // =====================
              // apply BCs and accumulate

              specfem::point::boundary<boundary_tag, dimension_tag, false>
                  point_boundary;
              specfem::assembly::load_on_device(index, boundaries,
                                                point_boundary);
              if constexpr (boundary_tag == specfem::element::boundary_tag::
                                                acoustic_free_surface) {
                specfem::boundary_conditions::apply_boundary_conditions(
                    point_boundary, self_accel);
              }
              specfem::assembly::atomic_add_on_device(index, field, self_accel);
            });
      });
}
