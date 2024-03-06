#ifndef _DOMAIN_KERNELS_TPP
#define _DOMAIN_KERNELS_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/interface.hpp"
#include "domain/impl/receivers/interface.hpp"
#include "domain/impl/sources/interface.hpp"
#include "enumerations/interface.hpp"
#include "kernels.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "quadrature/interface.hpp"

namespace {
/// Struct to tag each element
struct element_tag {

  element_tag(const specfem::element::medium_tag &medium_tag,
              const specfem::element::property_tag &property_tag,
              const specfem::element::boundary_tag_container &boundary_tag)
      : medium_tag(medium_tag), property_tag(property_tag),
        boundary_tag(boundary_tag) {}

  element_tag() = default;

  specfem::element::property_tag property_tag;
  specfem::element::boundary_tag_container boundary_tag;
  specfem::element::medium_tag medium_tag;
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium_tag,
          specfem::element::property_tag property_tag,
          specfem::element::boundary_tag boundary_tag, typename qp_type>
void allocate_elements(
    const specfem::compute::assembly &assembly, qp_type quadrature_points,
    const specfem::kokkos::HostView1d<element_tag> element_tags,
    specfem::domain::impl::kernels::element_kernel<DimensionType, medium_tag,
                                                   property_tag, boundary_tag,
                                                   qp_type> &elements) {

  using dimension = specfem::dimension::dimension<DimensionType>;
  using medium_type =
      specfem::medium::medium<DimensionType, medium_tag, property_tag>;
  using boundary_conditions_type =
      specfem::boundary::boundary<DimensionType, medium_tag, property_tag,
                                  boundary_tag, qp_type>;

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

  // // assert that boundary_conditions ispec matches with calculated ispec
  // if constexpr (((boundary_tag == specfem::enums::element::boundary_tag::
  //                                     acoustic_free_surface) &&
  //                (medium_tag == specfem::enums::element::type::acoustic))) {
  //   ASSERT(
  //       nelements == boundary_conditions.acoustic_free_surface.nelements,
  //       "nelements = " << nelements << " nelem_acoustic_surface = "
  //                      <<
  //                      boundary_conditions.acoustic_free_surface.nelements);
  //   for (int i = 0; i < nelements; i++) {
  //     ASSERT(h_ispec_domain(i) ==
  //                boundary_conditions.acoustic_free_surface.h_ispec(i),
  //            "Error: computing ispec for acoustic free surface elements");
  //   }
  // }

  // // assert that boundary_conditions ispec matches with calculated ispec
  // if constexpr ((boundary_tag ==
  //                specfem::enums::element::boundary_tag::stacey) &&
  //               (medium_tag == specfem::enums::element::type::acoustic)) {
  //   ASSERT(nelements == boundary_conditions.stacey.acoustic.nelements,
  //          "nelements = " << nelements << " nelements = "
  //                         << boundary_conditions.stacey.acoustic.nelements);
  //   for (int i = 0; i < nelements; i++) {
  //     ASSERT(h_ispec_domain(i) ==
  //                boundary_conditions.stacey.acoustic.h_ispec(i),
  //            "Error: computing ispec for stacey elements");
  //   }
  // } else if constexpr ((boundary_tag ==
  //                       specfem::enums::element::boundary_tag::stacey) &&
  //                      (medium_tag ==
  //                      specfem::enums::element::type::elastic)) {
  //   ASSERT(nelements == boundary_conditions.stacey.elastic.nelements,
  //          "nelements = " << nelements << " nelements = "
  //                         << boundary_conditions.stacey.elastic.nelements);
  //   for (int i = 0; i < nelements; i++) {
  //     ASSERT(h_ispec_domain(i) ==
  //     boundary_conditions.stacey.elastic.h_ispec(i),
  //            "Error: computing ispec for stacey elements");
  //   }
  // }

  // // assert that boundary_conditions ispec matches with calculated ispec
  // if constexpr ((boundary_tag == specfem::enums::element::boundary_tag::
  //                                    composite_stacey_dirichlet) &&
  //               (medium_tag == specfem::enums::element::type::acoustic)) {
  //   ASSERT(nelements ==
  //              boundary_conditions.composite_stacey_dirichlet.nelements,
  //          "nelements = "
  //              << nelements << " nelements = "
  //              << boundary_conditions.composite_stacey_dirichlet.nelements);
  //   for (int i = 0; i < nelements; i++) {
  //     ASSERT(h_ispec_domain(i) ==
  //                boundary_conditions.composite_stacey_dirichlet.h_ispec(i),
  //            "Error: computing ispec for stacey dirichlet elements");
  //   }
  // }

  // Copy ispec_domain to device
  // Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  std::cout << "  - Element type: \n"
            << "    - dimension           : " << dimension::to_string() << "\n"
            << "    - Element type        : " << medium_type::to_string()
            << "\n"
            << "    - Boundary Conditions : "
            << boundary_conditions_type::to_string() << "\n"
            << "    - Number of elements  : " << nelements << "\n\n";

  // Create isotropic acoustic surface elements
  elements = specfem::domain::impl::kernels::element_kernel<
      DimensionType, medium_tag, property_tag, boundary_tag, qp_type>(
      assembly, h_ispec_domain, quadrature_points);
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium_tag,
          specfem::element::property_tag property_tag, typename qp_type>
void allocate_isotropic_sources(
    const specfem::compute::assembly &assembly, qp_type quadrature_points,
    specfem::domain::impl::kernels::source_kernel<
        DimensionType, medium_tag, property_tag, qp_type> &isotropic_sources) {

  const auto value = medium_tag;
  // Create isotropic sources
  const auto ispec_array = assembly.sources.h_ispec_array;

  // Count the number of sources within this medium
  int nsources = 0;
  for (int isource = 0; isource < ispec_array.extent(0); isource++) {
    const int ispec = ispec_array(isource);
    if (assembly.properties.h_element_types(ispec) == value) {
      nsources++;
    }
  }

  // Save the index for sources in this domain
  specfem::kokkos::HostView1d<int> h_source_kernel_index_mapping(
      "specfem::domain::domain::source_kernel_index_mapping", nsources);

  specfem::kokkos::HostMirror1d<int> h_source_mapping(
      "specfem::domain::domain::source_mapping", nsources);

  int index = 0;
  for (int isource = 0; isource < ispec_array.extent(0); isource++) {
    const int ispec = ispec_array(isource);
    if (assembly.properties.h_element_types(ispec) == value) {
      h_source_kernel_index_mapping(index) = ispec_array(isource);
      h_source_mapping(index) = isource;
      index++;
    }
  }

  // Allocate isotropic sources
  isotropic_sources =
      specfem::domain::impl::kernels::source_kernel<DimensionType, medium_tag,
                                                    property_tag, qp_type>(
          assembly, h_source_kernel_index_mapping, h_source_mapping,
          quadrature_points);

  return;
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium_tag,
          specfem::element::property_tag property_tag, typename qp_type>
void allocate_isotropic_receivers(
    const specfem::compute::assembly &assembly, qp_type quadrature_points,
    specfem::domain::impl::kernels::receiver_kernel<DimensionType, medium_tag,
                                                    property_tag, qp_type>
        &isotropic_receivers) {

  const auto value = medium_tag;

  // Create isotropic sources
  const auto ispec_array = assembly.receivers.h_ispec_array;

  // Count the number of sources within this medium
  int nreceivers = 0;
  for (int ireceiver = 0; ireceiver < ispec_array.extent(0); ireceiver++) {
    const int ispec = ispec_array(ireceiver);
    if (assembly.properties.h_element_types(ispec) == value) {
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
    if (assembly.properties.h_element_types(ispec) == value) {
      h_receiver_kernel_index_mapping(index) = ispec_array(ireceiver);
      h_receiver_mapping(index) = ireceiver;
      index++;
    }
  }

  // Allocate isotropic sources
  isotropic_receivers =
      specfem::domain::impl::kernels::receiver_kernel<DimensionType, medium_tag,
                                                      property_tag, qp_type>(
          assembly, h_receiver_kernel_index_mapping, h_receiver_mapping,
          quadrature_points);

  return;
}
} // namespace

template <specfem::simulation::type simulation,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium, typename qp_type>
specfem::domain::impl::kernels::kernels<
    simulation, DimensionType, medium,
    qp_type>::kernels(const specfem::compute::assembly &assembly,
                      const qp_type &quadrature_points) {

  using medium_type = specfem::medium::medium<DimensionType, medium>;

  const int nspec = assembly.mesh.nspec;
  specfem::kokkos::HostView1d<element_tag> element_tags(
      "specfem::domain::domain::element_tag", nspec);
  // -----------------------------------------------------------
  // Start by tagging different elements
  // -----------------------------------------------------------
  // creating a context here for memory management
  // {
  //   // find medium type for every element
  //   specfem::kokkos::HostView1d<specfem::enums::element::type> ielement_type(
  //       "specfem::domain::impl::kernels::kernels::ielement_type", nspec);

  //   for (int ispec = 0; ispec < nspec; ispec++) {
  //     ielement_type(ispec) = properties.h_ispec_type(ispec);
  //   }

  //   // at start we consider every element is isotropic
  //   specfem::kokkos::HostView1d<specfem::enums::element::property_tag>
  //       ielement_property(
  //           "specfem::domain::impl::kernels::kernels::ielement_property",
  //           nspec);

  //   for (int ispec = 0; ispec < nspec; ispec++) {
  //     ielement_property(ispec) =
  //         specfem::enums::element::property_tag::isotropic;
  //   }

  //   // at start we consider every element is not on the boundary
  //   specfem::kokkos::HostView1d<specfem::enums::element::boundary_tag_container>
  //       ielement_boundary(
  //           "specfem::domain::impl::kernels::kernels::ielement_boundary",
  //           nspec);

  //   const auto &stacey = boundary_conditions.stacey;
  //   // mark stacey elements
  //   if (stacey.nelements > 0) {
  //     if (stacey.acoustic.nelements > 0) {
  //       for (int i = 0; i < stacey.acoustic.nelements; i++) {
  //         const int ispec = stacey.acoustic.h_ispec(i);
  //         ielement_boundary(ispec) +=
  //             specfem::enums::element::boundary_tag::stacey;
  //       }
  //     }

  //     if (stacey.elastic.nelements > 0) {
  //       for (int i = 0; i < stacey.elastic.nelements; i++) {
  //         const int ispec = stacey.elastic.h_ispec(i);
  //         ielement_boundary(ispec) +=
  //             specfem::enums::element::boundary_tag::stacey;
  //       }
  //     }
  //   }

  //   const auto &acoustic_free_surface =
  //       boundary_conditions.acoustic_free_surface;

  //   // mark acoustic free surface elements
  //   if (acoustic_free_surface.nelements > 0) {
  //     for (int i = 0; i < acoustic_free_surface.nelements; i++) {
  //       const int ispec = acoustic_free_surface.h_ispec(i);
  //       ielement_boundary(ispec) +=
  //           specfem::enums::element::boundary_tag::acoustic_free_surface;
  //     }
  //   }

  //   const auto &composite_stacey_dirichlet =
  //       boundary_conditions.composite_stacey_dirichlet;

  //   // mark composite stacey dirichlet elements
  //   if (composite_stacey_dirichlet.nelements > 0) {
  //     for (int i = 0; i < composite_stacey_dirichlet.nelements; i++) {
  //       const int ispec = composite_stacey_dirichlet.h_ispec(i);
  //       ielement_boundary(ispec) +=
  //           specfem::enums::element::boundary_tag::composite_stacey_dirichlet;
  //     }
  //   }

  //   // mark every element type
  //   for (int ispec = 0; ispec < nspec; ispec++) {
  //     element_tags(ispec) =
  //         element_tag(ielement_type(ispec), ielement_property(ispec),
  //                     ielement_boundary(ispec));
  //   }
  // }

  // -----------------------------------------------------------
  for (int ispec = 0; ispec < nspec; ispec++) {
    element_tags(ispec) =
        element_tag(assembly.properties.h_element_types(ispec),
                    assembly.properties.h_element_property(ispec),
                    assembly.boundaries.h_boundary_tags(ispec));
  }

  std::cout << " Element Statistics \n"
            << "------------------------------\n"
            << "- Types of elements in " << medium_type::to_string()
            << " medium :\n\n";

  // -----------------------------------------------------------

  // Allocate isotropic elements with dirichlet boundary conditions
  allocate_elements(assembly, quadrature_points, element_tags,
                    isotropic_elements_dirichlet);

  // Allocate isotropic elements with stacey boundary conditions
  allocate_elements(assembly, quadrature_points, element_tags,
                    isotropic_elements_stacey);

  // Allocate isotropic elements with stacey dirichlet boundary conditions
  allocate_elements(assembly, quadrature_points, element_tags,
                    isotropic_elements_stacey_dirichlet);

  // Allocate isotropic elements
  allocate_elements(assembly, quadrature_points, element_tags,
                    isotropic_elements);

  // Allocate isotropic sources

  allocate_isotropic_sources(assembly, quadrature_points, isotropic_sources);

  // Allocate isotropic receivers

  allocate_isotropic_receivers(assembly, quadrature_points,
                               isotropic_receivers);

  // Compute mass matrices

  this->compute_mass_matrix();

  return;
}

#endif // _DOMAIN_KERNELS_TPP
