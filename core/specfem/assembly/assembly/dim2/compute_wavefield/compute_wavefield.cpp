#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/assembly/impl/helper.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <type_traits>

namespace {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void get_wavefield_on_entire_grid(
    const specfem::enums::wavefield component,
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                 Kokkos::DefaultExecutionSpace>
        wavefield_on_entire_grid) {

  const auto &element_grid = assembly.mesh.element_grid;

  if (element_grid == 5) {
    specfem::assembly::assembly_impl::helper<
        specfem::element::dimension_tag::dim2, MediumTag, PropertyTag, 5>
        helper(assembly, wavefield_on_entire_grid);
    helper(component);
  } else if (element_grid == 8) {
    specfem::assembly::assembly_impl::helper<
        specfem::element::dimension_tag::dim2, MediumTag, PropertyTag, 8>
        helper(assembly, wavefield_on_entire_grid);
    helper(component);
  } else {
    throw std::runtime_error("Number of quadrature points not supported");
  }

  return;
}

} // namespace

Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
specfem::assembly::assembly<specfem::element::dimension_tag::dim2>::
    generate_wavefield_on_entire_grid(
        const specfem::simulation::field_type wavefield,
        const specfem::enums::wavefield component) {

  // Check which type of wavefield component is requested
  const int ncomponents = [&]() -> int {
    if (component == specfem::enums::wavefield::displacement) {
      return 2;
    } else if (component == specfem::enums::wavefield::velocity) {
      return 2;
    } else if (component == specfem::enums::wavefield::acceleration) {
      return 2;
    } else if (component == specfem::enums::wavefield::pressure) {
      return 1;
    } else if (component == specfem::enums::wavefield::rotation) {
      return 2;
    } else if (component == specfem::enums::wavefield::intrinsic_rotation) {
      return 2;
    } else if (component == specfem::enums::wavefield::curl) {
      return 2;
    } else {
      throw std::runtime_error("Wavefield component not supported");
    }
  }();

  // Copy the required wavefield into the buffer
  if (wavefield == specfem::simulation::field_type::forward) {
    specfem::assembly::deep_copy(this->fields.buffer, this->fields.forward);
  } else if (wavefield == specfem::simulation::field_type::adjoint) {
    specfem::assembly::deep_copy(this->fields.buffer, this->fields.adjoint);
  } else if (wavefield == specfem::simulation::field_type::backward) {
    specfem::assembly::deep_copy(this->fields.buffer, this->fields.backward);
  } else {
    throw std::runtime_error("Wavefield type not supported");
  }

  // Creates a view to store the wavefield on the entire grid
  Kokkos::View<type_real ****, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      wavefield_on_entire_grid("wavefield_on_entire_grid", this->mesh.nspec,
                               this->mesh.element_grid.ngllz,
                               this->mesh.element_grid.ngllx, ncomponents);

  // Create host mirror for the wavefield on the entire grid
  const auto h_wavefield_on_entire_grid =
      Kokkos::create_mirror_view(wavefield_on_entire_grid);

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
          ATTENUATION_SET(none),
      [&]<typename TagsType>() {
        get_wavefield_on_entire_grid<TagsType::medium_tag,
                                     TagsType::property_tag>(
            component, *this, wavefield_on_entire_grid);
      });

  // Copy the wavefield on the entire grid to the host
  Kokkos::deep_copy(h_wavefield_on_entire_grid, wavefield_on_entire_grid);

  return h_wavefield_on_entire_grid;
}
