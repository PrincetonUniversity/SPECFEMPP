#include "../test_fixture/test_fixture.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "generate_data.hpp"
#include "point/coordinates.hpp"
#include "point/field.hpp"
#include "test_helper.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template <specfem::wavefield::component component>
void test_element_wavefield(
    const int ispec,
    const Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &wavefield,
    specfem::compute::assembly &assembly) {

  const auto properties = assembly.properties;

  const auto medium = properties.h_element_types(ispec);
  const auto property = properties.h_element_property(ispec);

  if ((medium == specfem::element::medium_tag::elastic) &&
      (property == specfem::element::property_tag::isotropic)) {
    test_helper<component, specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic>
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

template <specfem::wavefield::component component,
          specfem::wavefield::type type>
void test_compute_wavefield(specfem::compute::assembly &assembly) {

  const auto ispecs = generate_data<component, type>(assembly);

  const auto wavefield =
      assembly.generate_wavefield_on_entire_grid(type, component);

  for (const auto ispec : ispecs) {
    test_element_wavefield<component>(ispec, wavefield, assembly);
  }
}

void test_compute_wavefield(specfem::compute::assembly &assembly) {

  try {
    test_compute_wavefield<specfem::wavefield::component::displacement,
                           specfem::wavefield::type::forward>(assembly);
  } catch (std::exception &e) {
    std::ostringstream message;
    message << "Error in computing displacement wavefield: \n\t" << e.what();
    throw std::runtime_error(message.str());
  }

  try {
    test_compute_wavefield<specfem::wavefield::component::velocity,
                           specfem::wavefield::type::forward>(assembly);
  } catch (std::exception &e) {
    std::ostringstream message;
    message << "Error in computing displacement wavefield: \n\t" << e.what();
    throw std::runtime_error(message.str());
  }

  try {
    test_compute_wavefield<specfem::wavefield::component::acceleration,
                           specfem::wavefield::type::forward>(assembly);
  } catch (std::exception &e) {
    std::ostringstream message;
    message << "Error in computing displacement wavefield: \n\t" << e.what();
    throw std::runtime_error(message.str());
  }
}

TEST_F(ASSEMBLY, compute_wavefield) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    auto assembly = std::get<1>(parameters);

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