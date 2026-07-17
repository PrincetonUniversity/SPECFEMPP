#include "specfem/linear_system/element_stiffness.tpp"

#include "specfem/element/to_string.hpp"
#include "specfem/tags.hpp"
#include <sstream>
#include <stdexcept>

namespace specfem::linear_system_impl {
/// Tag bundle for the only combination explicitly instantiated for the
/// linear system (issue #1982).
using elastic_isotropic_tags =
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
} // namespace specfem::linear_system_impl

template <typename Tags>
void specfem::linear_system::validate_stiffness_scope(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {

  static_assert(Tags::dimension_tag == specfem::element::dimension_tag::dim3,
                "validate_stiffness_scope takes a dim3 assembly; Tags must be "
                "a dim3 bundle.");

  if (assembly.mesh.element_grid != 5) {
    throw std::runtime_error(
        "specfem::linear_system::validate_stiffness_scope: only NGLL == 5 "
        "meshes are supported (the only 3D instantiation).");
  }

  const auto &element_types = assembly.element_types;
  const auto elements = element_types.get_elements_on_host(Tags::medium_tag);

  for (int i = 0; i < elements.size(); ++i) {
    const int ispec = elements(i);

    if (element_types.get_property_tag(ispec) != Tags::property_tag) {
      std::ostringstream message;
      message << "specfem::linear_system::validate_stiffness_scope: element "
              << ispec << " has property tag '"
              << specfem::element::to_string(
                     element_types.get_property_tag(ispec))
              << "', expected '"
              << specfem::element::to_string(Tags::property_tag) << "'.";
      throw std::runtime_error(message.str());
    }

    if (element_types.get_attenuation_tag(ispec) != Tags::attenuation_tag) {
      std::ostringstream message;
      message << "specfem::linear_system::validate_stiffness_scope: element "
              << ispec << " has attenuation tag '"
              << specfem::element::to_string(
                     element_types.get_attenuation_tag(ispec))
              << "'; only '"
              << specfem::element::to_string(Tags::attenuation_tag)
              << "' is supported.";
      throw std::runtime_error(message.str());
    }

    const auto boundary_tag = element_types.get_boundary_tag(ispec);
    if (boundary_tag != specfem::element::boundary_tag::none &&
        boundary_tag != specfem::element::boundary_tag::acoustic_free_surface) {
      std::ostringstream message;
      message << "specfem::linear_system::validate_stiffness_scope: element "
              << ispec << " has boundary tag '"
              << specfem::element::to_string(boundary_tag)
              << "'; only natural boundary conditions ('none', "
                 "'acoustic_free_surface') are supported. Stacey boundaries "
                 "add a velocity-dependent term the stiffness probe does not "
                 "capture.";
      throw std::runtime_error(message.str());
    }
  }
}

template <typename Tags>
void specfem::linear_system::compute_element_stiffness(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e) {
  static_assert(Tags::dimension_tag == specfem::element::dimension_tag::dim3,
                "compute_element_stiffness takes a dim3 assembly; Tags must "
                "be a dim3 bundle.");
  if (assembly.mesh.element_grid == 5) {
    specfem::linear_system::compute_element_stiffness<5, Tags>(assembly, batch,
                                                               k_e);
    return;
  }
  throw std::runtime_error(
      "specfem::linear_system::compute_element_stiffness: only NGLL == 5 is "
      "instantiated for 3D meshes.");
}

// Explicit instantiations: 3D elastic isotropic, NGLL = 5
template void specfem::linear_system::validate_stiffness_scope<
    specfem::linear_system_impl::elastic_isotropic_tags>(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &);

template void specfem::linear_system::compute_element_stiffness<
    5, specfem::linear_system_impl::elastic_isotropic_tags>(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const specfem::datatype::ElementIndexRange &,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &);

template void specfem::linear_system::compute_element_stiffness<
    specfem::linear_system_impl::elastic_isotropic_tags>(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const specfem::datatype::ElementIndexRange &,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &);
