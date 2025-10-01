#include "../test_fixture/test_fixture.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "generate_data.hpp"
#include "specfem/point.hpp"
#include "test_helper.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template <specfem::wavefield::type component>
void test_element_wavefield(
    const int ispec,
    const Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &wavefield,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  const auto element_types = assembly.element_types;

  const auto medium = element_types.get_medium_tag(ispec);
  const auto property = element_types.get_property_tag(ispec);

  if ((medium == specfem::element::medium_tag::elastic_psv) &&
      (property == specfem::element::property_tag::isotropic)) {
    test_helper<component, specfem::element::medium_tag::elastic_psv,
                specfem::element::property_tag::isotropic>
        handle(ispec, wavefield, assembly);
    handle.test();
  } else if ((medium == specfem::element::medium_tag::elastic_psv) &&
             (property == specfem::element::property_tag::anisotropic)) {
    test_helper<component, specfem::element::medium_tag::elastic_psv,
                specfem::element::property_tag::anisotropic>
        handle(ispec, wavefield, assembly);
    handle.test();
  } else if ((medium == specfem::element::medium_tag::elastic_sh) &&
             (property == specfem::element::property_tag::isotropic)) {
    test_helper<component, specfem::element::medium_tag::elastic_sh,
                specfem::element::property_tag::isotropic>
        handle(ispec, wavefield, assembly);
    handle.test();
  } else if ((medium == specfem::element::medium_tag::elastic_sh) &&
             (property == specfem::element::property_tag::anisotropic)) {
    test_helper<component, specfem::element::medium_tag::elastic_sh,
                specfem::element::property_tag::anisotropic>
        handle(ispec, wavefield, assembly);
    handle.test();
  } else if ((medium == specfem::element::medium_tag::acoustic) &&
             (property == specfem::element::property_tag::isotropic)) {
    test_helper<component, specfem::element::medium_tag::acoustic,
                specfem::element::property_tag::isotropic>
        handle(ispec, wavefield, assembly);
    handle.test();
  } else {
    throw std::runtime_error("Unsupported medium and property combination");
  }
}

template <specfem::wavefield::type component,
          specfem::wavefield::simulation_field type>
void test_compute_wavefield(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  const auto ispecs = generate_data<component, type>(assembly);

  /// You cannot generate a pressure field within an SH medium
  /// ----------------------------------
  const auto sh_ispec = assembly.element_types.get_elements_on_host(
      specfem::element::medium_tag::elastic_sh);

  if (component == specfem::wavefield::type::pressure &&
      (sh_ispec.extent(0) != 0)) {
    return;
  }
  /// ----------------------------------

  const auto wavefield =
      assembly.generate_wavefield_on_entire_grid(type, component);

  for (const auto ispec : ispecs) {
    test_element_wavefield<component>(ispec, wavefield, assembly);
  }
}

void test_compute_wavefield(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  try {
    test_compute_wavefield<specfem::wavefield::type::displacement,
                           specfem::wavefield::simulation_field::forward>(
        assembly);
  } catch (std::exception &e) {
    std::ostringstream message;
    message << "Error in computing displacement wavefield: \n\t" << e.what();
    throw std::runtime_error(message.str());
  }

  try {
    test_compute_wavefield<specfem::wavefield::type::velocity,
                           specfem::wavefield::simulation_field::forward>(
        assembly);
  } catch (std::exception &e) {
    std::ostringstream message;
    message << "Error in computing velocity wavefield: \n\t" << e.what();
    throw std::runtime_error(message.str());
  }

  try {
    test_compute_wavefield<specfem::wavefield::type::acceleration,
                           specfem::wavefield::simulation_field::forward>(
        assembly);
  } catch (std::exception &e) {
    std::ostringstream message;
    message << "Error in computing acceleration wavefield: \n\t" << e.what();
    throw std::runtime_error(message.str());
  }

  try {
    test_compute_wavefield<specfem::wavefield::type::pressure,
                           specfem::wavefield::simulation_field::forward>(
        assembly);
  } catch (std::exception &e) {
    std::ostringstream message;
    message << "Error in computing pressure wavefield: \n\t" << e.what();
    throw std::runtime_error(message.str());
  }
}

TEST_F(Assembly2D, compute_wavefield) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    specfem::assembly::assembly<specfem::dimension::type::dim2> assembly =
        std::get<5>(parameters);

    try {
      test_compute_wavefield(assembly);

      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m " << Test.name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
    } catch (std::exception &e) {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;31m[FAILED]\033[0m \n"
                << "-------------------------------------------------------\n"
                << "- Test: " << Test.name << "\n"
                << "- Error: " << e.what() << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      ADD_FAILURE();
    }
  }
}
