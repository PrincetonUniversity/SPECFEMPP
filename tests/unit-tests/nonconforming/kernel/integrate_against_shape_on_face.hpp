#pragma once
#include "acoustic_elastic.hpp"
#include "specfem/assembly/element_intersections/dim3/element_intersections.hpp"
#include "specfem/execution.hpp"
#include "specfem/mesh_entity/dim3/mesh_entity.hpp"
#include "specfem/point/jacobian_matrix.hpp"
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace specfem::nonconforming_test::kernel {

namespace impl {

template <typename OutputFieldViewType, typename InputFieldViewType>
KOKKOS_INLINE_FUNCTION void _atomic_add_iglob_for_all_extents(
    const OutputFieldViewType &outfield, const InputFieldViewType &infield,
    const int &iglob, const type_real &weight_jac) {
  if constexpr (InputFieldViewType::rank == 1) {
    Kokkos::atomic_add(&outfield(iglob), weight_jac * infield(iglob));
  } else if constexpr (InputFieldViewType::rank == 2) {
    for (int i1 = 0; i1 < infield.extent(1); i1++) {
      Kokkos::atomic_add(&outfield(iglob, i1), weight_jac * infield(iglob, i1));
    }
  } else {
    static_assert(InputFieldViewType::rank == 1,
                  "specfem::nonconforming_test::kernel::impl::integrate_"
                  "against_shape_on_faces: unsupported rank extent.");
  }
}

template <specfem::element::medium_tag medium_tag, typename OutputFieldType,
          typename InputFieldViewType>
OutputFieldType integrate_against_shape_on_faces(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace> &faces,
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &h_faces,
    const InputFieldViewType &dof_field) {
  constexpr specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim3;

  const auto field = assembly.fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();
  const int nglob = field.template get_nglob<medium_tag>();

  if (nglob != dof_field.extent(0)) {
    std::ostringstream oss;
    oss << "specfem::nonconforming_test::kernel::integrate_against_shape_on_"
           "faces: the field to integrate must have the same degrees of "
           "freedom";
    throw std::runtime_error(oss.str());
  }

  OutputFieldType output_field("output_field", nglob);
  Kokkos::deep_copy(output_field, 0);

  using parallel_config =
      specfem::parallel_configuration::default_chunk_face_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>;
  specfem::execution::ChunkedFaceIterator chunk(parallel_config(), faces);

  Kokkos::View<type_real ***> jacobian2d("jacobian2d", faces.N, faces.n_points,
                                         faces.n_points);
  const decltype(jacobian2d)::host_mirror_type h_jacobian2d =
      Kokkos::create_mirror_view(jacobian2d);

  for (int iface = 0; iface < h_faces.N; iface++) {
    const auto face = h_faces(iface);
    for (int ipoint = 0; ipoint < h_faces.n_points; ipoint++) {
      for (int jpoint = 0; jpoint < h_faces.n_points; jpoint++) {
        const auto index = face(ipoint, jpoint);
        specfem::point::jacobian_matrix<dimension_tag, true /*StoreJacobian*/,
                                        false /*UseSIMD*/>
            jac;

        specfem::assembly::load_on_host(index, assembly.jacobian_matrix, jac);
        const auto normal = jac.compute_normal(index.face_type);
        h_jacobian2d(iface, ipoint, jpoint) =
            std::sqrt(normal(0) * normal(0) + normal(1) * normal(1) +
                      normal(2) * normal(2));
      }
    }
  }

  Kokkos::deep_copy(jacobian2d, h_jacobian2d);
  specfem::execution::for_all(
      "specfem::compute::impl::compute_coupling", chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();

        const int iglob = field.template get_iglob<true, medium_tag>(index);

        type_real weight_jac =
            jacobian2d(index.iface, index.ipoint_i, index.ipoint_j) *
            assembly.mesh.weights(index.ipoint_i) *
            assembly.mesh.weights(index.ipoint_j);

        // An extended __host__ __device__ lambda cannot first-capture variable
        // in constexpr-if context
        _atomic_add_iglob_for_all_extents(output_field, dof_field, iglob,
                                          weight_jac);
      });
  Kokkos::fence();
  return output_field;
}
} // namespace impl

/**
 * @brief Integrates a scalar field along the given faces.
 */
template <specfem::element::medium_tag medium_tag, typename InputFieldViewType>
Kokkos::View<type_real *> integrate_against_shape_on_faces(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace> &faces,
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &h_faces,
    const InputFieldViewType &dof_field)
  requires(InputFieldViewType::rank == 1)
{
  return impl::integrate_against_shape_on_faces<medium_tag,
                                                Kokkos::View<type_real *>>(
      assembly, faces, h_faces, dof_field);
}

/**
 * @brief Integrates a vector field along the given faces.
 */
template <specfem::element::medium_tag medium_tag, typename InputFieldViewType>
Kokkos::View<type_real *[InputFieldViewType::static_extent(1)]>
integrate_against_shape_on_faces(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace> &faces,
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &h_faces,
    const InputFieldViewType &dof_field)
  requires(InputFieldViewType::rank == 2 &&
           InputFieldViewType::static_extent(1) != Kokkos::dynamic_extent)
{
  return impl::integrate_against_shape_on_faces<
      medium_tag,
      Kokkos::View<type_real * [InputFieldViewType::static_extent(1)]>>(
      assembly, faces, h_faces, dof_field);
}

template <typename FaceViewType>
std::pair<specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>,
          specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::
              host_mirror_type> collect_unique_faces(const FaceViewType
                                                         &self_faces,
                                                     const int &ngllz,
                                                     const int &nglly,
                                                     const int &ngllx) {

  std::vector<specfem::mesh_entity::face<specfem::element::dimension_tag::dim3>>
      collected_faces;
  std::map<std::pair<int, specfem::mesh_entity::dim3::type>, int> face_indices;
  int num_self_faces = 0;

  // populate this enumeration
  for (int iface = 0; iface < self_faces.N; ++iface) {
    const auto key = std::make_pair<int, specfem::mesh_entity::dim3::type>(
        self_faces(iface).element_index, self_faces(iface).face_type);
    if (face_indices.find(key) == face_indices.end()) {
      // face_indices does not yet have this (ispec, self_face_type)
      face_indices[key] = num_self_faces;
      collected_faces.push_back(
          { key.first, key.second, num_self_faces, false });
      num_self_faces++;
    }
  }
  specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace> faces(
      "self_faces_on_intersection", num_self_faces, ngllz);
  decltype(faces)::host_mirror_type h_faces =
      specfem::assembly::face_view_from_collected_faces(
          "h_self_faces_on_intersection", collected_faces,
          specfem::mesh_entity::element<specfem::element::dimension_tag::dim3>(
              ngllz, nglly, ngllx));
  specfem::assembly::deep_copy(faces, h_faces);
  return std::make_pair(faces, h_faces);
}
template <specfem::element_coupling::interface_tag interface_tag>
std::pair<specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>,
          specfem::assembly::FaceView<
              Kokkos::DefaultExecutionSpace>::host_mirror_type>
get_self_faces_on_intersection(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  const auto [self_faces, coupled_faces] =
      assembly.element_intersections.get_intersections_on_host(
          specfem::element_connections::type::nonconforming, interface_tag,
          specfem::element::boundary_tag::none,
          specfem::element_coupling::flux_scheme_tag::natural);

  const int ngllz = assembly.mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::ngllz;
  const int nglly = assembly.mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::nglly;
  const int ngllx = assembly.mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::ngllx;

  return collect_unique_faces(self_faces, ngllz, nglly, ngllx);
}
template <specfem::element_coupling::interface_tag interface_tag>
std::pair<specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>,
          specfem::assembly::FaceView<
              Kokkos::DefaultExecutionSpace>::host_mirror_type>
get_coupled_faces_on_intersection(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  const auto [self_faces, coupled_faces] =
      assembly.element_intersections.get_intersections_on_host(
          specfem::element_connections::type::nonconforming, interface_tag,
          specfem::element::boundary_tag::none,
          specfem::element_coupling::flux_scheme_tag::natural);

  const int ngllz = assembly.mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::ngllz;
  const int nglly = assembly.mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::nglly;
  const int ngllx = assembly.mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::ngllx;

  return collect_unique_faces(coupled_faces, ngllz, nglly, ngllx);
}

} // namespace specfem::nonconforming_test::kernel
