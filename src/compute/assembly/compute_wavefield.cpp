
#include "compute/assembly/assembly.hpp"
#include "helper.hpp"

namespace {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void get_wavefield_on_entire_grid(
    const specfem::wavefield::type component,
    const specfem::compute::assembly &assembly,
    Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                 Kokkos::DefaultExecutionSpace>
        wavefield_on_entire_grid) {

  const int ngllx = assembly.mesh.ngllx;
  const int ngllz = assembly.mesh.ngllz;

  if (ngllx == 5 && ngllz == 5) {
    impl::helper<MediumTag, PropertyTag, 5> helper(assembly,
                                                   wavefield_on_entire_grid);
    helper(component);
  } else if (ngllx == 8 && ngllz == 8) {
    impl::helper<MediumTag, PropertyTag, 8> helper(assembly,
                                                   wavefield_on_entire_grid);
    helper(component);
  } else {
    throw std::runtime_error("Number of quadrature points not supported");
  }

  return;
}

} // namespace

Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
specfem::compute::assembly::generate_wavefield_on_entire_grid(
    const specfem::wavefield::simulation_field wavefield,
    const specfem::wavefield::type component) {

  // Check which type of wavefield component is requested
  const int ncomponents = [&]() -> int {
    if (component == specfem::wavefield::type::displacement) {
      return 2;
    } else if (component == specfem::wavefield::type::velocity) {
      return 2;
    } else if (component == specfem::wavefield::type::acceleration) {
      return 2;
    } else if (component == specfem::wavefield::type::pressure) {
      return 1;
    } else {
      throw std::runtime_error("Wavefield component not supported");
    }
  }();

  // Copy the required wavefield into the buffer
  if (wavefield == specfem::wavefield::simulation_field::forward) {
    specfem::compute::deep_copy(this->fields.buffer, this->fields.forward);
  } else if (wavefield == specfem::wavefield::simulation_field::adjoint) {
    specfem::compute::deep_copy(this->fields.buffer, this->fields.adjoint);
  } else if (wavefield == specfem::wavefield::simulation_field::backward) {
    specfem::compute::deep_copy(this->fields.buffer, this->fields.backward);
  } else {
    throw std::runtime_error("Wavefield type not supported");
  }

  // Creates a view to store the wavefield on the entire grid
  Kokkos::View<type_real ****, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      wavefield_on_entire_grid("wavefield_on_entire_grid", this->mesh.nspec,
                               this->mesh.ngllz, this->mesh.ngllx, ncomponents);

  // Create host mirror for the wavefield on the entire grid
  const auto h_wavefield_on_entire_grid =
      Kokkos::create_mirror_view(wavefield_on_entire_grid);

  // Define a macro to call the get_wavefield_on_entire_grid function
  // for each material system
#define CALL_GET_WAVEFIELD_ON_ENTIRE_GRID_FUNCTION(DIMENSION_TAG, MEDIUM_TAG,  \
                                                   PROPERTY_TAG)               \
  if constexpr (GET_TAG(DIMENSION_TAG) == specfem::dimension::type::dim2) {    \
    get_wavefield_on_entire_grid<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(  \
        component, *this, wavefield_on_entire_grid);                           \
  }

  // Call the get_wavefield_on_entire_grid function for each material system
  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      CALL_GET_WAVEFIELD_ON_ENTIRE_GRID_FUNCTION,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH,
                MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef CALL_GET_WAVEFIELD_ON_ENTIRE_GRID_FUNCTION

  // get_wavefield_on_entire_grid<specfem::element::medium_tag::elastic_sv,
  //                              specfem::element::property_tag::isotropic>(
  //     component, *this, wavefield_on_entire_grid);

  // get_wavefield_on_entire_grid<specfem::element::medium_tag::elastic_sv,
  //                              specfem::element::property_tag::anisotropic>(
  //     component, *this, wavefield_on_entire_grid);

  // get_wavefield_on_entire_grid<specfem::element::medium_tag::acoustic,
  //                              specfem::element::property_tag::isotropic>(
  //     component, *this, wavefield_on_entire_grid);

  // Copy the wavefield on the entire grid to the host
  Kokkos::deep_copy(h_wavefield_on_entire_grid, wavefield_on_entire_grid);

  return h_wavefield_on_entire_grid;
}
