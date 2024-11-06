#include "compute/interface.hpp"
#include "kokkos_abstractions.h"
#include "material/material.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

namespace {
void compute_number_of_elements_per_medium(
    const int nspec, const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags &tags,
    const specfem::kokkos::HostView1d<specfem::element::medium_tag>
        &h_element_types,
    const specfem::kokkos::HostView1d<specfem::element::property_tag>
        &h_element_property,
    int &n_elastic, int &n_acoustic) {

  Kokkos::parallel_reduce(
      "specfem::compute::properties::compute_number_of_elements_per_medium",
      specfem::kokkos::HostRange(0, nspec),
      [=](const int ispec, int &n_elastic, int &n_acoustic) {
        const int ispec_mesh = mapping.compute_to_mesh(ispec);
        if (tags.tags_container(ispec_mesh).medium_tag ==
            specfem::element::medium_tag::elastic) {
          n_elastic++;
          h_element_types(ispec) = specfem::element::medium_tag::elastic;
          if (tags.tags_container(ispec_mesh).property_tag ==
              specfem::element::property_tag::isotropic) {
            h_element_property(ispec) =
                specfem::element::property_tag::isotropic;
          } else {
            throw std::runtime_error("Unknown property tag");
          }
        } else if (tags.tags_container(ispec_mesh).medium_tag ==
                   specfem::element::medium_tag::acoustic) {
          n_acoustic++;
          h_element_types(ispec) = specfem::element::medium_tag::acoustic;
          if (tags.tags_container(ispec_mesh).property_tag ==
              specfem::element::property_tag::isotropic) {
            h_element_property(ispec) =
                specfem::element::property_tag::isotropic;
          } else {
            throw std::runtime_error("Unknown property tag");
          }
        }
      },
      n_elastic, n_acoustic);

  if (n_elastic + n_acoustic != nspec)
    throw std::runtime_error("Number of elements per medium does not match "
                             "total number of elements");

  return;
}
} // namespace

specfem::compute::properties::properties(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags &tags, const specfem::mesh::materials &materials)
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

  compute_number_of_elements_per_medium(nspec, mapping, tags, h_element_types,
                                        h_element_property, n_elastic,
                                        n_acoustic);

  acoustic_isotropic = specfem::compute::impl::properties::material_property<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>(
      nspec, n_acoustic, ngllz, ngllx, mapping, tags, materials,
      h_property_index_mapping);

  elastic_isotropic = specfem::compute::impl::properties::material_property<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>(nspec, n_elastic, ngllz, ngllx,
                                                 mapping, tags, materials,
                                                 h_property_index_mapping);

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
  Kokkos::deep_copy(element_types, h_element_types);
  Kokkos::deep_copy(element_property, h_element_property);

  return;
}

Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
specfem::compute::properties::get_elements_on_host(
    const specfem::element::medium_tag medium) const {
  const auto &elements = [&]() -> impl::elements_of_type & {
    if (medium == specfem::element::medium_tag::elastic) {
      return elastic_elements;
    } else if (medium == specfem::element::medium_tag::acoustic) {
      return acoustic_elements;
    } else {
      throw std::runtime_error("Unknown medium tag");
    }
  }();

  if (elements.computed)
    return elements.h_elements;

  const int nspec = this->nspec;

  int nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (element_types(ispec) == medium) {
      nelements++;
    }
  }

  elements.elements =
      Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>(
          "specfem::compute::properties::get_elements_on_host", nelements);

  elements.h_elements = Kokkos::create_mirror_view(elements.elements);

  nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (element_types(ispec) == medium) {
      elements.h_elements(nelements) = ispec;
      nelements++;
    }
  }

  Kokkos::deep_copy(elements.elements, elements.h_elements);

  elements.computed = true;

  return elements.h_elements;
}

Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
get_elements_on_device(const specfem::element::medium_tag medium) const {
  const auto &elements = [&]() {
    if (medium == specfem::element::medium_tag::elastic) {
      return elastic_elements;
    } else if (medium == specfem::element::medium_tag::acoustic) {
      return acoustic_elements;
    } else {
      throw std::runtime_error("Unknown medium tag");
    }
  }();

  if (elements.computed)
    return elements.elements;

  // If the elements have not been computed, compute them.
  // The elements need to be computed in serial on the host.
  // This function computes the host elements on host and then
  // copies them to the device.
  const auto dummy = this->get_elements_on_host(medium);

  return elements.elements;
}

Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
specfem::compute::properties::get_elements_on_host(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property) const {
  const auto &elements = [&]() -> impl::elements_of_type & {
    if (medium == specfem::element::medium_tag::elastic) {
      if (property == specfem::element::property_tag::isotropic) {
        return elastic_isotropic_elements;
      } else {
        throw std::runtime_error("Unknown property tag");
      }
    } else if (medium == specfem::element::medium_tag::acoustic) {
      if (property == specfem::element::property_tag::isotropic) {
        return acoustic_isotropic_elements;
      } else {
        throw std::runtime_error("Unknown property tag");
      }
    } else {
      throw std::runtime_error("Unknown medium tag");
    }
  }();

  if (elements.computed)
    return elements.h_elements;

  const int nspec = this->nspec;

  int nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (element_types(ispec) == medium && element_property(ispec) == property) {
      nelements++;
    }
  }

  elements.elements =
      Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>(
          "specfem::compute::properties::get_elements_on_host", nelements);

  elements.h_elements = Kokkos::create_mirror_view(elements.elements);

  nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (element_types(ispec) == medium && element_property(ispec) == property) {
      elements.h_elements(nelements) = ispec;
      nelements++;
    }
  }

  Kokkos::deep_copy(elements.elements, elements.h_elements);

  elements.computed = true;

  return elements.h_elements;
}

Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
get_elements_on_device(const specfem::element::medium_tag medium,
                       const specfem::element::property_tag property) const {
  const auto &elements = [&]() {
    if (medium == specfem::element::medium_tag::elastic) {
      if (property == specfem::element::property_tag::isotropic) {
        return elastic_isotropic_elements;
      } else {
        throw std::runtime_error("Unknown property tag");
      }
    } else if (medium == specfem::element::medium_tag::acoustic) {
      if (property == specfem::element::property_tag::isotropic) {
        return acoustic_isotropic_elements;
      } else {
        throw std::runtime_error("Unknown property tag");
      }
    } else {
      throw std::runtime_error("Unknown medium tag");
    }
  }();

  if (elements.computed)
    return elements.elements;

  // If the elements have not been computed, compute them.
  // The elements need to be computed in serial on the host.
  // This function computes the host elements on host and then
  // copies them to the device.
  const auto dummy = this->get_elements_on_host(medium, property);

  return elements.elements;
}
