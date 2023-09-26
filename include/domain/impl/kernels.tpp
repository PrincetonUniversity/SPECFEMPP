#ifndef _DOMAIN_KERNELS_TPP
#define _DOMAIN_KERNELS_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"

template <class medium, class qp_type>
specfem::domain::impl::kernels::kernels<medium, qp_type>::kernels(
    const specfem::kokkos::DeviceView3d<int> ibool,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::properties &properties,
    const specfem::compute::sources &sources,
    quadrature::quadrature *quadx, quadrature::quadrature *quadz,
    qp_type quadrature_points,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> mass_matrix) {

  const int nspec = partial_derivatives.xix.extent(0);
  const auto value = medium::value;

  // count number of elements in this domain
  int nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (properties.h_ispec_type(ispec) == value) {
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
    if (properties.h_ispec_type(ispec) == value) {
      h_ispec_domain(index) = ispec;
      index++;
    }
  }

  // Copy ispec_domain to device
  Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  // Create isotropic elements
  isotropic_elements = specfem::domain::impl::kernels::element_kernel<
      medium_type, quadrature_point_type,
      specfem::enums::element::property::isotropic>(
      ibool, ispec_domain, partial_derivatives, properties, quadx, quadz,
      quadrature_points, field, field_dot_dot, mass_matrix);

  // Create isotropic sources

  const auto ispec_array = sources.h_ispec_array;
  int nsources = 0;
  for (int isource = 0; isource < ispec_array.extent(0); isource++) {
    if (properties.h_ispec_type(ispec_array(isource)) == value) {
      nsources++;
    }
  }

  specfem::kokkos::DeviceView1d<int> ispec_sources(
      "specfem::domain::domain::ispec_sources", nsources);

  specfem::kokkos::HostMirror1d<int> h_ispec_sources =
      Kokkos::create_mirror_view(ispec_sources);

  specfem::kokkos::DeviceView1d<int> isource_array(
      "specfem::domain::domain::isource_array", nsources);

  specfem::kokkos::HostMirror1d<int> h_isource_array =
      Kokkos::create_mirror_view(isource_array);

  index = 0;
  for (int isource = 0; isource < ispec_array.extent(0); isource++) {
    if (properties.h_ispec_type(ispec_array(isource)) == value) {
      h_ispec_sources(index) = ispec_array(isource);
      h_isource_array(index) = isource;
      index++;
    }
  }

  Kokkos::deep_copy(ispec_sources, h_ispec_sources);
  Kokkos::deep_copy(isource_array, h_isource_array);

  isotropic_sources = specfem::domain::impl::kernels::source_kernel<
      medium_type, quadrature_point_type,
      specfem::enums::element::property::isotropic>(
      ibool, ispec_sources, isource_array, properties, sources, quadrature_points, field_dot_dot);


  // Create isotropic receivers

  return;
}

#endif // _DOMAIN_KERNELS_TPP
