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

  element_tag(
      const specfem::enums::element::type &medium_tag,
      const specfem::enums::element::property_tag &property_tag,
      const specfem::enums::element::boundary_tag_container &boundary_tag)
      : medium_tag(medium_tag), property_tag(property_tag),
        boundary_tag(boundary_tag) {}

  element_tag() = default;

  specfem::enums::element::property_tag property_tag;
  specfem::enums::element::boundary_tag_container boundary_tag;
  specfem::enums::element::type medium_tag;
};

template <class medium, class qp_type, class property, class BC>
void allocate_elements(
    const specfem::compute::assembly &assembly, qp_type quadrature_points,
    const specfem::kokkos::HostView1d<element_tag> element_tags,
    specfem::domain::impl::kernels::element_kernel<medium, qp_type, property,
                                                   BC> &elements) {

  constexpr auto boundary_tag = BC::value;
  constexpr auto medium_tag = medium::value;
  constexpr auto property_tag = property::value;

  const int nspec = assembly.mesh.nspec;

  // count number of elements in this domain
  int nelements = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (element_tags(ispec).medium_tag == medium_tag &&
        element_tags(ispec).property_tag == property_tag &&
        element_tags(ispec).boundary_tag == boundary_tag) {

      // make sure acoustic free surface elements are acoustic
      if (element_tags(ispec).boundary_tag ==
          specfem::enums::element::boundary_tag::acoustic_free_surface) {
        if (element_tags(ispec).medium_tag !=
            specfem::enums::element::type::acoustic) {
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
  Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  std::cout << "  - Element type: \n"
            << "    - dimension           : "
            << specfem::enums::element::dimension::dim2::to_string() << "\n"
            << "    - property            : " << property::to_string() << "\n"
            << "    - Boundary Conditions : " << BC::to_string() << "\n"
            << "    - Number of elements  : " << nelements << "\n\n";

  // Create isotropic acoustic surface elements
  elements = specfem::domain::impl::kernels::element_kernel<medium, qp_type,
                                                            property, BC>(
      assembly, h_ispec_domain, quadrature_points);
}

template <class medium, class qp_type>
void allocate_isotropic_sources(
    const specfem::compute::assembly &assembly, qp_type quadrature_points,
    specfem::domain::impl::kernels::source_kernel<
        medium, qp_type, specfem::enums::element::property::isotropic>
        &isotropic_sources) {

  const auto value = medium::value;

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
  isotropic_sources = specfem::domain::impl::kernels::source_kernel<
      medium, qp_type, specfem::enums::element::property::isotropic>(
      assembly, h_source_kernel_index_mapping, h_source_mapping,
      quadrature_points);

  return;
}

// template <class medium, class qp_type>
// void allocate_isotropic_receivers(
//     const specfem::kokkos::DeviceView3d<int> ibool,
//     const specfem::compute::partial_derivatives &partial_derivatives,
//     const specfem::compute::properties &properties,
//     const specfem::compute::receivers &receivers,
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
//     field_dot_dot, specfem::quadrature::quadrature *quadx,
//     specfem::quadrature::quadrature *quadz, qp_type quadrature_points,
//     specfem::domain::impl::kernels::receiver_kernel<
//         medium, qp_type, specfem::enums::element::property::isotropic>
//         &isotropic_receivers) {

//   const auto value = medium::value;

//   // Create isotropic receivers

//   const auto ispec_array = receivers.h_ispec_array;
//   int nreceivers = 0;
//   for (int ireceiver = 0; ireceiver < ispec_array.extent(0); ireceiver++) {
//     if (properties.h_ispec_type(ispec_array(ireceiver)) == value) {
//       nreceivers++;
//     }
//   }

//   specfem::kokkos::DeviceView1d<int> ispec_receivers(
//       "specfem::domain::domain::ispec_receivers", nreceivers);

//   specfem::kokkos::HostMirror1d<int> h_ispec_receivers =
//       Kokkos::create_mirror_view(ispec_receivers);

//   specfem::kokkos::DeviceView1d<int> ireceiver_array(
//       "specfem::domain::domain::ireceiver_array", nreceivers);

//   specfem::kokkos::HostMirror1d<int> h_ireceiver_array =
//       Kokkos::create_mirror_view(ireceiver_array);

//   int index = 0;
//   for (int ireceiver = 0; ireceiver < ispec_array.extent(0); ireceiver++) {
//     if (properties.h_ispec_type(ispec_array(ireceiver)) == value) {
//       h_ispec_receivers(index) = ispec_array(ireceiver);
//       h_ireceiver_array(index) = ireceiver;
//       index++;
//     }
//   }

//   Kokkos::deep_copy(ispec_receivers, h_ispec_receivers);
//   Kokkos::deep_copy(ireceiver_array, h_ireceiver_array);

//   isotropic_receivers = specfem::domain::impl::kernels::receiver_kernel<
//       medium, qp_type, specfem::enums::element::property::isotropic>(
//       ibool, ispec_receivers, ireceiver_array, partial_derivatives,
//       properties, receivers, field, field_dot, field_dot_dot, quadx, quadz,
//       quadrature_points);

//   return;
// }
} // namespace

template <class medium, class qp_type>
specfem::domain::impl::kernels::kernels<medium, qp_type>::kernels(
    const specfem::compute::assembly &assembly,
    const qp_type &quadrature_points) {

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
            << "- Types of elements in " << medium::to_string()
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

  // // Allocate isotropic sources

  allocate_isotropic_sources(assembly, quadrature_points, isotropic_sources);

  // // Allocate isotropic receivers

  // allocate_isotropic_receivers(assembly, quadrature_points,
  //                              isotropic_receivers);

  return;
}

#endif // _DOMAIN_KERNELS_TPP
