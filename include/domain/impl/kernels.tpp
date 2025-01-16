#pragma once

#include "boundary_conditions/boundary_conditions.hpp"
#include "kernels.hpp"

namespace {
/// Struct to tag each element
struct element_tag {

  element_tag(const specfem::element::medium_tag &medium_tag,
              const specfem::element::property_tag &property_tag,
              const specfem::element::boundary_tag &boundary_tag)
      : medium_tag(medium_tag), property_tag(property_tag),
        boundary_tag(boundary_tag) {}

  element_tag() = default;

  specfem::element::property_tag property_tag;
  specfem::element::boundary_tag boundary_tag;
  specfem::element::medium_tag medium_tag;
};

template <typename ElementType>
void allocate_elements(
    const specfem::compute::assembly &assembly,
    const specfem::kokkos::HostView1d<element_tag> element_tags,
    ElementType &elements) {

  constexpr auto wavefield_type = ElementType::wavefield_type;
  constexpr auto medium_tag = ElementType::medium_tag;
  constexpr auto property_tag = ElementType::property_tag;
  constexpr auto boundary_tag = ElementType::boundary_tag;

  using dimension = specfem::dimension::dimension<ElementType::dimension>;

  const auto element_indices = assembly.element_types.get_elements_on_host(
      medium_tag, property_tag, boundary_tag);

  if constexpr (wavefield_type ==
                    specfem::wavefield::simulation_field::forward ||
                wavefield_type ==
                    specfem::wavefield::simulation_field::adjoint) {

    std::cout << "  - Element type: \n"
              << "    - dimension           : " << dimension::to_string()
              << "\n"
              << "    - Element type        : "
              << specfem::element::to_string(medium_tag, property_tag,
                                             boundary_tag)
              << "\n"
              << "    - Number of elements  : " << element_indices.extent(0)
              << "\n\n";
  }

  // Create isotropic acoustic surface elements
  elements = { assembly, element_indices };
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium_tag,
          specfem::element::property_tag property_tag, int NGLL>
void allocate_isotropic_sources(const specfem::compute::assembly &assembly,
                                specfem::domain::impl::kernels::source_kernel<
                                    WavefieldType, DimensionType, medium_tag,
                                    property_tag, NGLL> &isotropic_sources) {

  // Allocate isotropic sources
  isotropic_sources = specfem::domain::impl::kernels::source_kernel<
      WavefieldType, DimensionType, medium_tag, property_tag, NGLL>(assembly);

  return;
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium_tag,
          specfem::element::property_tag property_tag, int NGLL>
void allocate_receivers(const specfem::compute::assembly &assembly,
                        specfem::domain::impl::kernels::receiver_kernel<
                            WavefieldType, DimensionType, medium_tag,
                            property_tag, NGLL> &receivers) {
  receivers = specfem::domain::impl::kernels::receiver_kernel<
      WavefieldType, DimensionType, medium_tag, property_tag, NGLL>(assembly);

  return;
}
} // namespace

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType, typename qp_type>
specfem::domain::impl::kernels::kernels<
    WavefieldType, DimensionType, specfem::element::medium_tag::elastic,
    qp_type>::kernels(const type_real dt,
                      const specfem::compute::assembly &assembly,
                      const qp_type &quadrature_points)
    : isotropic_receivers(assembly), anisotropic_receivers(assembly) {

  const int nspec = assembly.mesh.nspec;
  specfem::kokkos::HostView1d<element_tag> element_tags(
      "specfem::domain::domain::element_tag", nspec);

  // -----------------------------------------------------------
  for (int ispec = 0; ispec < nspec; ispec++) {
    element_tags(ispec) =
        element_tag(assembly.element_types.get_medium_tag(ispec),
                    assembly.element_types.get_property_tag(ispec),
                    assembly.element_types.get_boundary_tag(ispec));
  }

  if constexpr (WavefieldType ==
                    specfem::wavefield::simulation_field::forward ||
                WavefieldType ==
                    specfem::wavefield::simulation_field::adjoint) {
    std::cout << " Element Statistics \n"
              << "------------------------------\n"
              << "- Types of elements in elastic medium :\n\n";
  }

  // -----------------------------------------------------------

  // Allocate isotropic elements with dirichlet boundary conditions
  // allocate_elements(assembly, element_tags, isotropic_elements_dirichlet);

  // Allocate aniostropic elements with dirichlet boundary conditions
  // allocate_elements(assembly, element_tags, anisotropic_elements_dirichlet);

  // Allocate isotropic elements with stacey boundary conditions
  allocate_elements(assembly, element_tags, isotropic_elements_stacey);

  // Allocate anisotropic elements with stacey boundary conditions
  allocate_elements(assembly, element_tags, anisotropic_elements_stacey);

  // Allocate isotropic elements with stacey dirichlet boundary conditions
  // allocate_elements(assembly, element_tags,
  //                   isotropic_elements_stacey_dirichlet);

  // Allocate anisotropic elements with stacey dirichlet boundary conditions
  // allocate_elements(assembly, element_tags,
  //                   anisotropic_elements_stacey_dirichlet);

  // Allocate isotropic elements
  allocate_elements(assembly, element_tags, isotropic_elements);

  // Allocate anisotropic elements
  allocate_elements(assembly, element_tags, anisotropic_elements);

  // Allocate isotropic sources

  allocate_isotropic_sources(assembly, isotropic_sources);
  // Compute mass matrices

  this->compute_mass_matrix(dt);

  return;
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType, typename qp_type>
specfem::domain::impl::kernels::kernels<
    WavefieldType, DimensionType, specfem::element::medium_tag::acoustic,
    qp_type>::kernels(const type_real dt,
                      const specfem::compute::assembly &assembly,
                      const qp_type &quadrature_points)
    : isotropic_receivers(assembly) {

  const int nspec = assembly.mesh.nspec;
  specfem::kokkos::HostView1d<element_tag> element_tags(
      "specfem::domain::domain::element_tag", nspec);

  // -----------------------------------------------------------
  for (int ispec = 0; ispec < nspec; ispec++) {
    element_tags(ispec) =
        element_tag(assembly.element_types.get_medium_tag(ispec),
                    assembly.element_types.get_property_tag(ispec),
                    assembly.element_types.get_boundary_tag(ispec));
  }

  if constexpr (WavefieldType ==
                    specfem::wavefield::simulation_field::forward ||
                WavefieldType ==
                    specfem::wavefield::simulation_field::adjoint) {
    std::cout << " Element Statistics \n"
              << "------------------------------\n"
              << "- Types of elements in acoustic medium :\n\n";
  }

  // -----------------------------------------------------------

  // Allocate isotropic elements with dirichlet boundary conditions
  allocate_elements(assembly, element_tags, isotropic_elements_dirichlet);

  // Allocate isotropic elements with stacey boundary conditions
  allocate_elements(assembly, element_tags, isotropic_elements_stacey);

  // Allocate isotropic elements with stacey dirichlet boundary conditions
  allocate_elements(assembly, element_tags,
                    isotropic_elements_stacey_dirichlet);

  // Allocate isotropic elements
  allocate_elements(assembly, element_tags, isotropic_elements);

  // Allocate isotropic sources

  allocate_isotropic_sources(assembly, isotropic_sources);

  // Compute mass matrices

  this->compute_mass_matrix(dt);

  return;
}
