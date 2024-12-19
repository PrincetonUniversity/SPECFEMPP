#include "compute/impl/value_containers.hpp"

namespace {
void compute_number_of_elements_per_medium(
    const int nspec, const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::kokkos::HostView1d<specfem::element::medium_tag>
        &h_medium_tags,
    const specfem::kokkos::HostView1d<specfem::element::property_tag>
        &h_property_tags,
    int &n_elastic_isotropic, int &n_elastic_anisotropic, int &n_acoustic) {

  Kokkos::parallel_reduce(
      "specfem::compute::properties::compute_number_of_elements_per_medium",
      specfem::kokkos::HostRange(0, nspec),
      [=](const int ispec, int &n_elastic_isotropic, int &n_elastic_anisotropic,
          int &n_acoustic) {
        const int ispec_mesh = mapping.compute_to_mesh(ispec);
        if (tags.tags_container(ispec_mesh).medium_tag ==
            specfem::element::medium_tag::elastic) {
          h_medium_tags(ispec) = specfem::element::medium_tag::elastic;
          if (tags.tags_container(ispec_mesh).property_tag ==
              specfem::element::property_tag::isotropic) {
            n_elastic_isotropic++;
            h_property_tags(ispec) =
                specfem::element::property_tag::isotropic;
          } else if (tags.tags_container(ispec_mesh).property_tag ==
                     specfem::element::property_tag::anisotropic) {
            n_elastic_anisotropic++;
            h_property_tags(ispec) =
                specfem::element::property_tag::anisotropic;
          } else {
            throw std::runtime_error("Unknown property tag");
          }
        } else if (tags.tags_container(ispec_mesh).medium_tag ==
                   specfem::element::medium_tag::acoustic) {
          n_acoustic++;
          h_medium_tags(ispec) = specfem::element::medium_tag::acoustic;
          if (tags.tags_container(ispec_mesh).property_tag ==
              specfem::element::property_tag::isotropic) {
            h_property_tags(ispec) =
                specfem::element::property_tag::isotropic;
          } else {
            throw std::runtime_error("Unknown property tag");
          }
        }
      },
      n_elastic_isotropic, n_elastic_anisotropic, n_acoustic);

  if (n_elastic_isotropic + n_elastic_anisotropic + n_acoustic != nspec)
    throw std::runtime_error("Number of elements per medium does not match "
                             "total number of elements");

  return;
}
} // namespace

template <template <specfem::element::medium_tag, specfem::element::property_tag>
          class containers_type>
specfem::compute::impl::value_containers<containers_type>::value_containers(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags)
    : specfem::compute::impl::element_types(nspec, ngllz, ngllx, mapping, tags) {
  // compute total number of elastic and acoustic spectral elements
  int n_elastic_isotropic;
  int n_elastic_anisotropic;
  int n_acoustic;

  compute_number_of_elements_per_medium(nspec, mapping, tags, h_medium_tags,
                                        h_property_tags, n_elastic_isotropic,
                                        n_elastic_anisotropic, n_acoustic);

  acoustic_isotropic = containers_type<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>(
      nspec, n_acoustic, ngllz, ngllx, mapping, tags, h_property_index_mapping);

  elastic_isotropic = containers_type<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>(nspec, n_elastic_isotropic,
                                                 ngllz, ngllx, mapping, tags,
                                                 h_property_index_mapping);

  elastic_anisotropic = containers_type<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic>(nspec, n_elastic_anisotropic,
                                                   ngllz, ngllx, mapping, tags,
                                                   h_property_index_mapping);

  return;
}

template class specfem::compute::impl::value_containers<specfem::compute::impl::kernels::material_kernels>;
