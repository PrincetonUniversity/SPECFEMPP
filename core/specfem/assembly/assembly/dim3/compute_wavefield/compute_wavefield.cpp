#include "enumerations/interface.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/assembly/impl/helper.hpp"
#include "specfem/macros.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <type_traits>

namespace {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void get_wavefield_on_entire_grid(
    const specfem::wavefield::type component,
    const specfem::assembly::assembly<specfem::dimension::type::dim3> &assembly,
    Kokkos::View<type_real *****, Kokkos::LayoutLeft,
                 Kokkos::DefaultExecutionSpace>
        wavefield_on_entire_grid) {

  const auto &element_grid = assembly.mesh.element_grid;

  if (element_grid == 5) {
    specfem::assembly::assembly_impl::helper<specfem::dimension::type::dim3,
                                             MediumTag, PropertyTag, 5>
        helper(assembly, wavefield_on_entire_grid);
    helper(component);
  } else if (element_grid == 8) {
    specfem::assembly::assembly_impl::helper<specfem::dimension::type::dim3,
                                             MediumTag, PropertyTag, 8>
        helper(assembly, wavefield_on_entire_grid);
    helper(component);
  } else {
    throw std::runtime_error("Number of quadrature points not supported");
  }

  return;
}

} // namespace

Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>
specfem::assembly::assembly<specfem::dimension::type::dim3>::
    generate_wavefield_on_entire_grid(
        const specfem::wavefield::simulation_field wavefield,
        const specfem::wavefield::type component) {

  // Check which type of wavefield component is requested
  const int ncomponents = [&]() -> int {
    if (component == specfem::wavefield::type::displacement) {
      return 3;
    } else if (component == specfem::wavefield::type::velocity) {
      return 3;
    } else if (component == specfem::wavefield::type::acceleration) {
      return 3;
    } else if (component == specfem::wavefield::type::pressure) {
      return 1;
    } else if (component == specfem::wavefield::type::rotation) {
      return 3;
    } else if (component == specfem::wavefield::type::intrinsic_rotation) {
      return 3;
    } else if (component == specfem::wavefield::type::curl) {
      return 3;
    } else {
      throw std::runtime_error("Wavefield component not supported");
    }
  }();

  // Copy the required wavefield into the buffer
  if (wavefield == specfem::wavefield::simulation_field::forward) {
    specfem::assembly::deep_copy(this->fields.buffer, this->fields.forward);
  } else if (wavefield == specfem::wavefield::simulation_field::adjoint) {
    specfem::assembly::deep_copy(this->fields.buffer, this->fields.adjoint);
  } else if (wavefield == specfem::wavefield::simulation_field::backward) {
    specfem::assembly::deep_copy(this->fields.buffer, this->fields.backward);
  } else {
    throw std::runtime_error("Wavefield type not supported");
  }

  // Creates a view to store the wavefield on the entire grid
  Kokkos::View<type_real *****, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      wavefield_on_entire_grid("wavefield_on_entire_grid", this->mesh.nspec,
                               this->mesh.element_grid.ngllz,
                               this->mesh.element_grid.nglly,
                               this->mesh.element_grid.ngllx, ncomponents);

  // Create host mirror for the wavefield on the entire grid
  const auto h_wavefield_on_entire_grid =
      Kokkos::create_mirror_view(wavefield_on_entire_grid);

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC)), {
        if constexpr (_dimension_tag_ == specfem::dimension::type::dim3) {
          get_wavefield_on_entire_grid<_medium_tag_, _property_tag_>(
              component, *this, wavefield_on_entire_grid);
        }
      })

  // Copy the wavefield on the entire grid to the host
  Kokkos::deep_copy(h_wavefield_on_entire_grid, wavefield_on_entire_grid);

  return h_wavefield_on_entire_grid;
}
