#include "specfem/assembly/boundaries/dim3/impl/injection/initialize_injection.hpp"

#include "specfem/assembly/boundaries/dim3/impl/injection/extract_injection_points.hpp"
#include "specfem/injection/injection_provider.hpp"
#include "specfem/injection/make_provider.hpp"

specfem::assembly::boundaries_impl::injection_boundary<
    specfem::element::dimension_tag::dim3>
specfem::assembly::boundaries_impl::make_injection_boundary(
    const specfem::injection::injection_inputs &inputs,
    const specfem::assembly::boundaries_impl::stacey<
        specfem::element::dimension_tag::dim3> &stacey_data,
    const Kokkos::View<int *, Kokkos::DefaultExecutionSpace>::host_mirror_type
        &h_stacey_index_mapping,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim3>
        &assembly_properties,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &assembly_element_types,
    type_real dt, int nstep) {
  specfem::assembly::boundaries_impl::injection_boundary<
      specfem::element::dimension_tag::dim3>
      result;

  if (!inputs.enabled) {
    return result;
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
    return result;
  }

  auto provider =
      specfem::injection::make_provider<specfem::element::dimension_tag::dim3>(
          inputs, points, dt, nstep);
  provider->initialize();

  Kokkos::deep_copy(point_index, h_point_index);

  result.provider = provider;
  result.point_index = point_index;
  result.h_point_index = h_point_index;
  result.number_of_points = points.size();
  return result;
}
