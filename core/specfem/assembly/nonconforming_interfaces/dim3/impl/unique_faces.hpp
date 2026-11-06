#pragma once

#include "specfem/assembly/element_intersections.hpp"

namespace specfem::assembly::nonconforming_interfaces_impl {
/**
 * @brief Collects the unique faces (ispec, side) in the given view.
 * @param self_faces a host view of the faces to collect
 * @param ngllz quadrure points in the local-z axis (optional: defaults to
 * self_faces.n_points)
 * @param nglly quadrure points in the local-y axis (optional: defaults to
 * self_faces.n_points)
 * @param ngllx quadrure points in the local-x axis (optional: defaults to
 * self_faces.n_points)
 * @return
 * specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::host_mirror_type
 * a host-view of the unique faces.
 */
specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::host_mirror_type
collect_unique_faces(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &self_faces,
    const int &ngllz, const int &nglly, const int &ngllx);
specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::host_mirror_type
collect_unique_faces(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &self_faces);

/**
 * @brief Computes the average length scale of the given faces.
 * @param self_faces A host-view of the (non-unique) faces to average.
 * @param mesh assembly::mesh instance that the faces belong to
 * @return type_real the length scale of the faces (averaged across unique
 * faces), taken to be the largest diagonal for each face.
 */
type_real mean_face_length_scale(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &self_faces,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);
} // namespace specfem::assembly::nonconforming_interfaces_impl
