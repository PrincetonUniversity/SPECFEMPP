#include "../accessor.hpp"
#include "extra_kernel.hpp"
#include "specfem/boundary_conditions.hpp"
#include "specfem/chunk_face.hpp"
#include "specfem/datatype/accessor_type.hpp"
#include "specfem/datatype/chunk_face_view.hpp"
#include "specfem/datatype/register_array.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/tags.hpp"

#include "specfem/compute/impl/compute_coupling_subkernel/conjugate_integral.tpp"

// template <int NGLL>
// struct StackStoredChunkFaceArray
//     : public specfem::datatype::RegisterArray<
//           typename specfem::datatype::simd<type_real,
//                                            false /*using_simd*/>::datatype,
//           Kokkos::extents<std::size_t, 1, NGLL, NGLL, 1>, Kokkos::layout_left> {

//   constexpr static bool using_simd =
//       false; ///< Use SIMD datatypes for the array. If false,
//              ///< std::is_same<value_type, base_type>::value is true
//   using base_type = specfem::datatype::RegisterArray<
//       typename specfem::datatype::simd<type_real, using_simd>::datatype,
//       Kokkos::extents<std::size_t, 1, NGLL, NGLL, 1>, Kokkos::layout_left>;
//   using simd = specfem::datatype::simd<type_real, using_simd>; ///< SIMD data
//                                                                ///< type
//   using value_type =
//       typename base_type::value_type;  ///< Value type used to store
//                                        ///< the elements of the array
//   constexpr static int components = 1; ///< Number of components of the
//                                        ///< vector
//   static constexpr int ngll = NGLL;

//   using base_type::base_type;
//   constexpr static auto accessor_type =
//       specfem::datatype::AccessorType::chunk_face;
// };

template <int NGLL, typename Tags>
void specfem::element_coupling::TMP_extra_kernel::compute_coupling_extra_kernel<
    NGLL, Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
        (Tags::interface_tag ==
             specfem::element_coupling::interface_tag::acoustic_elastic ||
         Tags::interface_tag ==
             specfem::element_coupling::interface_tag::elastic_acoustic)>>::
    execute(const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {
  constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr auto flux_scheme_tag =
      specfem::element_coupling::flux_scheme_tag::natural;
  constexpr static auto connection_tag =
      specfem::element_connections::type::nonconforming;
  constexpr static auto interface_tag = Tags::interface_tag;
  constexpr static auto boundary_tag = Tags::boundary_tag;
  constexpr static auto wavefield_tag = Tags::wavefield_tag;
  const auto &interface_container =
      assembly.nonconforming_interfaces.template get_interface_container<
          interface_tag, boundary_tag, connection_tag, flux_scheme_tag>();

  const auto &flux_scheme_data = interface_container.flux_scheme_data;

  // ==================================================================
  //                     symmetrizer
  // ==================================================================
  if (flux_scheme_data.should_symmetrize_coupling) {
    specfem::compute::impl::compute_coupling_conjugate_integral_nonconforming<
        NGLL, Tags>(assembly);
  }
  if (false) {
    constexpr specfem::element_coupling::interface_tag conjugate_interface_tag =
        (interface_tag ==
         specfem::element_coupling::interface_tag::acoustic_elastic)
            ? specfem::element_coupling::interface_tag::elastic_acoustic
            : specfem::element_coupling::interface_tag::acoustic_elastic;
    const auto [conjugate_coupled_intersections, conjugate_self_intersections] =
        assembly.element_intersections.get_intersections_on_device(
            connection_tag, conjugate_interface_tag, boundary_tag,
            flux_scheme_tag);

    if (conjugate_self_intersections.N == 0 &&
        conjugate_coupled_intersections.N == 0)
      return;

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
            dimension_tag, conjugate_interface_tag, boundary_tag,
            flux_scheme_tag, parallel_config::chunk_size, NGLL,
            NQuad_intersection>;

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
            specfem::element::attributes<dimension_tag,
                                         self_medium>::components,
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
                // init this point's (accumulated to self_accel) shape function
                specfem::compute::impl::StackStoredChunkFaceArray<NGLL> self_shape_fcn;
                for (int ipoint = 0; ipoint < NGLL; ipoint++) {
                  for (int jpoint = 0; jpoint < NGLL; jpoint++) {
                    for (int icomp = 0; icomp < ncomp_self; icomp++) {
                      // iface = 0 since only 1 is needed (accumulation only
                      // occurs on own iface)
                      //
                      // icomp = 0 since we have a
                      // vectorviewtype (ncomp = 1)
                      self_shape_fcn(0, ipoint, jpoint, 0) = 0;
                    }
                  }
                }
                self_shape_fcn(0, index.ipoint_i, index.ipoint_j, 0) = 1;

                // =====================
                // interpolate this shape function onto the coupled

                for (int ipoint_coupled = 0; ipoint_coupled < NGLL;
                     ipoint_coupled++) {
                  for (int jpoint_coupled = 0; jpoint_coupled < NGLL;
                       jpoint_coupled++) {
                    auto coupled_index = conjugate_coupled_intersections(
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
                    specfem::datatype::VectorPointViewType<type_real, 1,
                                                           false /*UseSIMD*/>
                        interpolated_field;
                    coupled_index.iface = 0;
                    specfem::algorithms::transfer_interpolate(
                        coupled_index, point_interface_data, self_shape_fcn,
                        interpolated_field);

                    // accumulate self_accel by medium_physics::compute_coupling
                    if constexpr (interface_tag ==
                                  specfem::element_coupling::interface_tag::
                                      acoustic_elastic) {
                      // negative sign, since the normal is in the opposite
                      // direction.
                      self_accel(0) -=
                          interpolated_field(0) *
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
                      self_accel(0) -=
                          interpolated_field(0) *
                          point_interface_data.face_factor *
                          (point_interface_data.face_normal(0) *
                           coupled_field(iface_local, ipoint_coupled,
                                         jpoint_coupled, 0));
                      self_accel(1) -=
                          interpolated_field(0) *
                          point_interface_data.face_factor *
                          (point_interface_data.face_normal(1) *
                           coupled_field(iface_local, ipoint_coupled,
                                         jpoint_coupled, 0));
                      self_accel(2) -=
                          interpolated_field(0) *
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
                specfem::assembly::atomic_add_on_device(index, field,
                                                        self_accel);
              });
        });
  }

  // ==================================================================
  //                     alpha - self-merging
  // ==================================================================
  {

    const int &num_faces = flux_scheme_data.self_faces.num_faces;
    if (num_faces == 0) {
      return;
    }
    const auto &ngll = flux_scheme_data.self_faces.face_view.n_points;

    constexpr auto self_medium =
        specfem::element_coupling::attributes<dimension_tag,
                                              interface_tag>::self_medium();
    constexpr auto ncomp_self =
        specfem::element::attributes<dimension_tag, self_medium>::components;
    using ChunkFieldView =
        specfem::chunk_element::displacement<1, NGLL, dimension_tag,
                                             self_medium, false>;
    using PointTags = specfem::tags::Tags<
        dimension_tag, self_medium, specfem::element::property_tag::isotropic,
        specfem::element::attenuation_tag::constant_isotropic,
        false /*using_simd*/>;
    using PointFieldView = specfem::point::displacement<PointTags>;
    using PointAccelView = specfem::point::acceleration<PointTags>;
    using QuadratureType = specfem::quadrature::lagrange_derivative<
        NGLL, dimension_tag,
        Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    const auto field =
        assembly.fields.template get_simulation_field<wavefield_tag>();
    // compute normals
    using team_handle = Kokkos::TeamPolicy<>::member_type;
    Kokkos::parallel_for(
        "compute_coupling_extra_kernel",
        Kokkos::TeamPolicy<>(num_faces, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(ChunkFieldView::shmem_size() +
                                                 QuadratureType::shmem_size())),
        KOKKOS_LAMBDA(const team_handle &team) {
          const int &iface = team.league_rank();
          const auto face = flux_scheme_data.self_faces.face_view(iface);

          ChunkFieldView element_field(team.team_scratch(0));
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, NGLL * NGLL * NGLL),
              [&](const int &ijk) {
                const int k = ijk % NGLL;
                const int ij = ijk / NGLL;
                const int j = ij % NGLL;
                const int i = ij / NGLL;

                specfem::point::index<dimension_tag> index(face.element_index,
                                                           k, j, i);
                PointFieldView point_field;
                specfem::assembly::load_on_device(index, field, point_field);
                for (int icomp = 0; icomp < ncomp_self; icomp++) {
                  element_field(0, index.iz, index.iy, index.ix, icomp) =
                      point_field(icomp);
                }
              });

          QuadratureType lagrange_derivative(team);
          specfem::assembly::load_on_device(team, assembly.mesh,
                                            lagrange_derivative);

          team.team_barrier();
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &ij) {
                const int i = ij / NGLL;
                const int j = ij % NGLL;
                const auto index = face(i, j);
                using datatype = typename ChunkFieldView::simd::datatype;

                datatype df_dxi[ncomp_self] = { 0.0 };
                datatype df_deta[ncomp_self] = { 0.0 };
                datatype df_dgamma[ncomp_self] = { 0.0 };
                specfem::point::jacobian_matrix<dimension_tag, false,
                                                false /*using_simd*/>
                    point_jacobian_matrix;
                specfem::assembly::load_on_device(
                    index, assembly.jacobian_matrix, point_jacobian_matrix);

                specfem::point::index<dimension_tag> regular_index(
                    0, index.iz, index.iy,
                    index.ix); // ispec should be zero right now, since
                               // element_gradient uses it as a chunk ielem
                               // index.
                const auto grad = specfem::algorithms::impl::element_gradient(
                    element_field, regular_index, point_jacobian_matrix,
                    lagrange_derivative, df_dxi, df_deta, df_dgamma);
                regular_index.ispec = index.ispec;

                specfem::point::properties<PointTags> point_property;
                specfem::assembly::load_on_device(
                    regular_index, assembly.properties, point_property);
                const auto stress =
                    specfem::medium_physics::compute_stress<PointTags>(
                        point_property, grad);

                PointAccelView accel;
                for (int icomp = 0; icomp < ncomp_self; icomp++) {
                  accel(icomp) = 0;
                  for (int idim = 0;
                       idim < specfem::element::dimension<dimension_tag>::dim;
                       idim++) {
                    // take T directly, since we do not need the local->global
                    // conversion that divergence takes.
                    accel(icomp) += flux_scheme_data.normal_times_weight(
                                        iface, i, j, idim) *
                                    stress.T(icomp, idim);
                  }
                }

                specfem::point::boundary<boundary_tag, dimension_tag, false>
                    point_boundary;
                specfem::assembly::load_on_device(index, assembly.boundaries,
                                                  point_boundary);
                if constexpr (boundary_tag == specfem::element::boundary_tag::
                                                  acoustic_free_surface) {
                  specfem::boundary_conditions::apply_boundary_conditions(
                      point_boundary, accel);
                }

                specfem::assembly::atomic_add_on_device(index, field, accel);
              });
        });
  }
}
