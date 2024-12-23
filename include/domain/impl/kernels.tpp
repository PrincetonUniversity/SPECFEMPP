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
  using medium_type =
      specfem::element::attributes<ElementType::dimension, medium_tag>;

  const int nspec = assembly.mesh.nspec;

  // count number of elements in this domain
  int nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (element_tags(ispec).medium_tag == medium_tag &&
        element_tags(ispec).property_tag == property_tag &&
        element_tags(ispec).boundary_tag == boundary_tag) {

      // make sure acoustic free surface elements are acoustic
      if (element_tags(ispec).boundary_tag ==
          specfem::element::boundary_tag::acoustic_free_surface) {
        if (element_tags(ispec).medium_tag !=
            specfem::element::medium_tag::acoustic) {
          throw std::runtime_error("Error: acoustic free surface boundary "
                                   "condition found non acoustic element");
        }
      }
      nelements++;
    }
  }

  specfem::kokkos::DeviceView1d<int> ispec_domain(
      "specfem::domain::domain::h_ispec_domain", nelements);
  specfem::kokkos::HostMirror1d<int> h_ispec_domain =
      Kokkos::create_mirror_view(ispec_domain);

  // Get ispec for each element in this domain
  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (element_tags(ispec).medium_tag == medium_tag &&
        element_tags(ispec).property_tag == property_tag &&
        element_tags(ispec).boundary_tag == boundary_tag) {
      h_ispec_domain(index) = ispec;
      index++;
    }
  }

  if constexpr (wavefield_type ==
                    specfem::wavefield::simulation_field::forward ||
                wavefield_type ==
                    specfem::wavefield::simulation_field::adjoint) {

    std::cout << "  - Element type: \n"
              << "    - dimension           : " << dimension::to_string()
              << "\n"
              << "    - Element type        : "
              << specfem::element::to_string(medium_tag, property_tag, boundary_tag) << "\n"
              // << "    - Boundary Conditions : "
              // << specfem::domain::impl::boundary_conditions::print_boundary_tag<
              //        boundary_tag>()
              << "\n"
              << "    - Number of elements  : " << nelements << "\n\n";
  }

  // Create isotropic acoustic surface elements
  elements = { assembly, h_ispec_domain };
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium_tag,
          specfem::element::property_tag property_tag, int NGLL>
void allocate_isotropic_sources(
    const specfem::compute::assembly &assembly,
    specfem::domain::impl::kernels::source_kernel<WavefieldType, DimensionType,
                                                  medium_tag, property_tag, NGLL> &isotropic_sources) {

  // Allocate isotropic sources
  isotropic_sources = specfem::domain::impl::kernels::source_kernel<
      WavefieldType, DimensionType, medium_tag, property_tag, NGLL>(
      assembly);

  return;
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium_tag,
          specfem::element::property_tag property_tag, typename qp_type>
void allocate_receivers(const specfem::compute::assembly &assembly,
                        qp_type quadrature_points,
                        specfem::domain::impl::kernels::receiver_kernel<
                            WavefieldType, DimensionType, medium_tag,
                            property_tag, qp_type> &receivers) {

  const auto medium = medium_tag;
  const auto property = property_tag;

  // Create isotropic sources
  const auto ispec_array = assembly.receivers.h_ispec_array;

  // Count the number of sources within this medium
  int nreceivers = 0;
  for (int ireceiver = 0; ireceiver < ispec_array.extent(0); ireceiver++) {
    const int ispec = ispec_array(ireceiver);
    if (assembly.properties.h_element_types(ispec) == medium &&
        assembly.properties.h_element_property(ispec) == property) {
      nreceivers++;
    }
  }

  // Save the index for sources in this domain
  specfem::kokkos::HostView1d<int> h_receiver_kernel_index_mapping(
      "specfem::domain::domain::receiver_kernel_index_mapping", nreceivers);

  specfem::kokkos::HostMirror1d<int> h_receiver_mapping(
      "specfem::domain::domain::receiver_mapping", nreceivers);

  int index = 0;
  for (int ireceiver = 0; ireceiver < ispec_array.extent(0); ireceiver++) {
    const int ispec = ispec_array(ireceiver);
    if (assembly.properties.h_element_types(ispec) == medium &&
        assembly.properties.h_element_property(ispec) == property) {
      h_receiver_kernel_index_mapping(index) = ispec_array(ireceiver);
      h_receiver_mapping(index) = ireceiver;
      index++;
    }
  }

  // Allocate isotropic sources
  receivers = specfem::domain::impl::kernels::receiver_kernel<
      WavefieldType, DimensionType, medium_tag, property_tag, qp_type>(
      assembly, h_receiver_kernel_index_mapping, h_receiver_mapping,
      quadrature_points);

  return;
}
} // namespace

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType, typename qp_type>
specfem::domain::impl::kernels::kernels<
    WavefieldType, DimensionType, specfem::element::medium_tag::elastic,
    qp_type>::kernels(const type_real dt,
                      const specfem::compute::assembly &assembly,
                      const qp_type &quadrature_points) {

  const int nspec = assembly.mesh.nspec;
  specfem::kokkos::HostView1d<element_tag> element_tags(
      "specfem::domain::domain::element_tag", nspec);

  // -----------------------------------------------------------
  for (int ispec = 0; ispec < nspec; ispec++) {
    element_tags(ispec) =
        element_tag(assembly.properties.h_element_types(ispec),
                    assembly.properties.h_element_property(ispec),
                    assembly.boundaries.boundary_tags(ispec));
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

  // Allocate isotropic receivers

  allocate_receivers(assembly, quadrature_points, isotropic_receivers);

  // Allocate anisotropic receivers

  allocate_receivers(assembly, quadrature_points, anisotropic_receivers);

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
                      const qp_type &quadrature_points) {

  const int nspec = assembly.mesh.nspec;
  specfem::kokkos::HostView1d<element_tag> element_tags(
      "specfem::domain::domain::element_tag", nspec);

  // -----------------------------------------------------------
  for (int ispec = 0; ispec < nspec; ispec++) {
    element_tags(ispec) =
        element_tag(assembly.properties.h_element_types(ispec),
                    assembly.properties.h_element_property(ispec),
                    assembly.boundaries.boundary_tags(ispec));
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

  // Allocate isotropic receivers

  allocate_receivers(assembly, quadrature_points,
                               isotropic_receivers);

  // Compute mass matrices

  this->compute_mass_matrix(dt);

  return;
}
