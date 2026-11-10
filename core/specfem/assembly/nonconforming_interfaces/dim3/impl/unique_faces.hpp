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
 * @brief Sum of characteristic face lengths and number of unique faces.
 */
struct FaceLengthTotal {
  type_real length_sum = 0; ///< sum over the unique faces
  int count = 0;            ///< number of unique faces
};

/**
 * @brief Sums the characteristic length of the unique faces in the given view.
 * @param faces A host-view of the (non-unique) faces to sum.
 * @param mesh assembly::mesh instance that the faces belong to
 * @return FaceLengthTotal the sum of the characteristic lengths (the largest
 * diagonal of each face) over the unique faces, and the number of unique faces.
 */
FaceLengthTotal face_length_sum(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &faces,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);
} // namespace specfem::assembly::nonconforming_interfaces_impl
