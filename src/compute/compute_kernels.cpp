#include "compute/kernels/kernels.hpp"

specfem::compute::kernels::kernels(const int nspec, const int ngllz,
                                   const int ngllx,
                                   const specfem::mesh::materials &materials)
    : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
      element_types("specfem::compute::properties::element_types", nspec),
      h_element_types(Kokkos::create_mirror_view(element_types)),
      property_index_mapping(
          "specfem::compute::properties::property_index_mapping", nspec),
      element_property("specfem::compute::properties::element_property", nspec),
      h_element_property(Kokkos::create_mirror_view(element_property)),
      h_property_index_mapping(
          Kokkos::create_mirror_view(property_index_mapping)) {
  // compute total number of elastic and acoustic spectral elements
  int n_elastic;
  int n_acoustic;

  Kokkos::parallel_reduce(
      "specfem::compute::kernels", this->nspec,
      KOKKOS_LAMBDA(const int ispec, int &n_elastic, int &n_acoustic) {
        if (materials.material_index_mapping(ispec).type ==
            specfem::element::medium_tag::elastic) {
          n_elastic++;
          h_element_types(ispec) = specfem::element::medium_tag::elastic;
          h_element_property(ispec) = specfem::element::property_tag::isotropic;
        } else if (materials.material_index_mapping(ispec).type ==
                   specfem::element::medium_tag::acoustic) {
          n_acoustic++;
          h_element_types(ispec) = specfem::element::medium_tag::acoustic;
          h_element_property(ispec) = specfem::element::property_tag::isotropic;
        }
      },
      n_elastic, n_acoustic);

  assert(n_elastic + n_acoustic == nspec);

  acoustic_isotropic = specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>(
      nspec, n_acoustic, ngllz, ngllx, materials, h_property_index_mapping);

  elastic_isotropic = specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>(
      nspec, n_elastic, ngllz, ngllx, materials, h_property_index_mapping);

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
  Kokkos::deep_copy(element_types, h_element_types);
  Kokkos::deep_copy(element_property, h_element_property);

  return;
}
