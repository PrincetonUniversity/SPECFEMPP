#include "specfem/assembly/boundaries/dim3/impl/injection/initialize_injection.hpp"

#include "specfem/assembly/boundaries/dim3/impl/injection/extract_injection_points.hpp"
#include "specfem/injection/injection_provider.hpp"
#include "specfem/injection/make_provider.hpp"

std::shared_ptr<specfem::injection::injection_provider<
    specfem::element::dimension_tag::dim3>>
specfem::assembly::boundaries_impl::make_injection_boundary(
    const specfem::injection::injection_inputs &inputs,
    const specfem::assembly::boundaries_impl::stacey<
        specfem::element::dimension_tag::dim3> &stacey_data,
    const Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
        &stacey_index_mapping,
    const Kokkos::View<int *, Kokkos::DefaultExecutionSpace>::host_mirror_type
        &h_stacey_index_mapping,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim3>
        &assembly_properties,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &assembly_element_types,
    type_real dt, int nstep,
    specfem::assembly::boundaries_impl::injection_boundary<
        specfem::element::dimension_tag::dim3> &out) {
  out = {};

  if (!inputs.enabled) {
    return nullptr;
  }

  using PointIndexView = specfem::assembly::boundaries_impl::injection_boundary<
      specfem::element::dimension_tag::dim3>::PointIndexView;
  const auto &tags = stacey_data.h_quadrature_point_boundary_tag;
  PointIndexView point_index("specfem::assembly::injection_point_index",
                             tags.extent(0), tags.extent(1), tags.extent(2),
                             tags.extent(3));
  auto h_point_index = Kokkos::create_mirror_view(point_index);

  const specfem::injection::injection_points points =
      specfem::assembly::boundaries_impl::extract_stacey_injection_points(
          stacey_data, h_stacey_index_mapping, assembly_mesh,
          assembly_properties, assembly_element_types, h_point_index);

  // No Stacey points: leave the injection boundary empty (effective no-op).
  if (points.size() == 0) {
    return nullptr;
  }

  // The Stacey injection consumer reads the incident velocity v* (frame slots
  // 0-2) to form the impedance term Z(v*); the vector quantity must be
  // velocity.
  specfem::injection::injection_inputs stacey_inputs = inputs;
  stacey_inputs.derivative = specfem::injection::fk::field_derivative::velocity;

  auto provider =
      specfem::injection::make_provider<specfem::element::dimension_tag::dim3>(
          stacey_inputs, points, dt, nstep);
  provider->initialize();

  Kokkos::deep_copy(point_index, h_point_index);

  out.point_index = point_index;
  out.h_point_index = h_point_index;
  out.stacey_index_mapping = stacey_index_mapping;
  out.frame = provider->frame_buffer();
  out.number_of_points = points.size();
  return provider;
}
