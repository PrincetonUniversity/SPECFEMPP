
#include "compute/assembly/assembly.hpp"
#include "helper.hpp"

namespace {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::wavefield::component Component>
void get_wavefield_on_entire_grid(
    specfem::compute::assembly &assembly,
    Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                 Kokkos::DefaultExecutionSpace>
        wavefield_on_entire_grid) {

  const int ngllx = assembly.mesh.ngllx;
  const int ngllz = assembly.mesh.ngllz;

  if (ngllx == 5 && ngllz == 5) {
    impl::helper<MediumTag, PropertyTag, Component, 5> helper(
        assembly, wavefield_on_entire_grid);
    helper();
  } else if (ngllx == 8 && ngllz == 8) {
    impl::helper<MediumTag, PropertyTag, Component, 8> helper(
        assembly, wavefield_on_entire_grid);
    helper();
  } else {
    throw std::runtime_error("Number of quadrature points not supported");
  }

  return;
}

} // namespace

Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
specfem::compute::assembly::generate_wavefield_on_entire_grid(
    const specfem::wavefield::type wavefield,
    const specfem::wavefield::component component) {

  const int ncomponents = [&]() -> int {
    if (component == specfem::wavefield::component::displacement) {
      return 2;
    } else if (component == specfem::wavefield::component::velocity) {
      return 2;
    } else if (component == specfem::wavefield::component::acceleration) {
      return 2;
    } else {
      throw std::runtime_error("Wavefield component not supported");
    }
  }();

  // Copy the required wavefield into the buffer
  if (wavefield == specfem::wavefield::type::forward) {
    specfem::compute::deep_copy(this->fields.buffer, this->fields.forward);
  } else if (wavefield == specfem::wavefield::type::adjoint) {
    specfem::compute::deep_copy(this->fields.buffer, this->fields.adjoint);
  } else if (wavefield == specfem::wavefield::type::backward) {
    specfem::compute::deep_copy(this->fields.buffer, this->fields.backward);
  } else {
    throw std::runtime_error("Wavefield type not supported");
  }

  Kokkos::View<type_real ****, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      wavefield_on_entire_grid("wavefield_on_entire_grid", this->mesh.nspec,
                               this->mesh.ngllz, this->mesh.ngllx, ncomponents);

  const auto h_wavefield_on_entire_grid =
      Kokkos::create_mirror_view(wavefield_on_entire_grid);

  if (component == specfem::wavefield::component::displacement) {
    get_wavefield_on_entire_grid<specfem::element::medium_tag::elastic,
                                 specfem::element::property_tag::isotropic,
                                 specfem::wavefield::component::displacement>(
        *this, wavefield_on_entire_grid);
    get_wavefield_on_entire_grid<specfem::element::medium_tag::acoustic,
                                 specfem::element::property_tag::isotropic,
                                 specfem::wavefield::component::displacement>(
        *this, wavefield_on_entire_grid);
  } else if (component == specfem::wavefield::component::velocity) {
    get_wavefield_on_entire_grid<specfem::element::medium_tag::elastic,
                                 specfem::element::property_tag::isotropic,
                                 specfem::wavefield::component::velocity>(
        *this, wavefield_on_entire_grid);
    get_wavefield_on_entire_grid<specfem::element::medium_tag::acoustic,
                                 specfem::element::property_tag::isotropic,
                                 specfem::wavefield::component::velocity>(
        *this, wavefield_on_entire_grid);
  } else if (component == specfem::wavefield::component::acceleration) {
    get_wavefield_on_entire_grid<specfem::element::medium_tag::elastic,
                                 specfem::element::property_tag::isotropic,
                                 specfem::wavefield::component::acceleration>(
        *this, wavefield_on_entire_grid);
    get_wavefield_on_entire_grid<specfem::element::medium_tag::acoustic,
                                 specfem::element::property_tag::isotropic,
                                 specfem::wavefield::component::acceleration>(
        *this, wavefield_on_entire_grid);
  } else {
    throw std::runtime_error("Wavefield component not supported");
  }

  Kokkos::deep_copy(h_wavefield_on_entire_grid, wavefield_on_entire_grid);

  return h_wavefield_on_entire_grid;
}
