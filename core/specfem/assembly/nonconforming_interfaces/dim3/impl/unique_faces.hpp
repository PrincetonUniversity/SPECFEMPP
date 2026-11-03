#pragma once

#include "specfem/assembly/element_intersections.hpp"

namespace specfem::assembly::nonconforming_interfaces_impl {
specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::host_mirror_type
collect_unique_faces(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &self_faces);
specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::host_mirror_type
collect_unique_faces(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &self_faces,
    const int &ngllz, const int &nglly, const int &ngllx);

type_real mean_face_length_scale(
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &self_faces,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);
} // namespace specfem::assembly::nonconforming_interfaces_impl
