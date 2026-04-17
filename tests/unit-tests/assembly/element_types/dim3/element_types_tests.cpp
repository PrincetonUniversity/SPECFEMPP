#include "specfem/assembly/element_types.hpp"
#include "specfem/element.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

using ElementTypes3D =
    specfem::assembly::element_types<specfem::element::dimension_tag::dim3>;

// Test 1: Homogeneous mesh - all elements elastic/isotropic/none/none
TEST(ElementTypes3D, HomogeneousMesh) {
  const int nspec = 4;
  const int ngllz = 5;
  const int nglly = 5;
  const int ngllx = 5;

  Kokkos::View<specfem::element::medium_tag *,
               Kokkos::DefaultHostExecutionSpace>
      medium_tags("medium_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    medium_tags(i) = specfem::element::medium_tag::elastic;
  }

  Kokkos::View<specfem::element::property_tag *,
               Kokkos::DefaultHostExecutionSpace>
      property_tags("property_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    property_tags(i) = specfem::element::property_tag::isotropic;
  }

  Kokkos::View<specfem::element::attenuation_tag *,
               Kokkos::DefaultHostExecutionSpace>
      attenuation_tags("attenuation_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    attenuation_tags(i) = specfem::element::attenuation_tag::none;
  }

  Kokkos::View<specfem::element::boundary_tag *,
               Kokkos::DefaultHostExecutionSpace>
      boundary_tags("boundary_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    boundary_tags(i) = specfem::element::boundary_tag::none;
  }

  ElementTypes3D element_types(nspec, ngllz, nglly, ngllx, medium_tags,
                               property_tags, attenuation_tags, boundary_tags);

  // Test number of elements
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::elastic),
            4);
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::acoustic),
            0);

  // Test host view
  auto elastic_host =
      element_types.get_elements_on_host(specfem::element::medium_tag::elastic);
  EXPECT_EQ(elastic_host.extent(0), 4);
  EXPECT_EQ(elastic_host(0), 0);
  EXPECT_EQ(elastic_host(1), 1);
  EXPECT_EQ(elastic_host(2), 2);
  EXPECT_EQ(elastic_host(3), 3);

  // Test device view
  auto elastic_device = element_types.get_elements_on_device(
      specfem::element::medium_tag::elastic);
  EXPECT_EQ(elastic_device.extent(0), 4);

  // Test per-element getters
  for (int i = 0; i < nspec; i++) {
    EXPECT_EQ(element_types.get_medium_tag(i),
              specfem::element::medium_tag::elastic);
    EXPECT_EQ(element_types.get_property_tag(i),
              specfem::element::property_tag::isotropic);
    EXPECT_EQ(element_types.get_attenuation_tag(i),
              specfem::element::attenuation_tag::none);
    EXPECT_EQ(element_types.get_boundary_tag(i),
              specfem::element::boundary_tag::none);
  }
}

// Test 2: Mixed media - elastic and acoustic
TEST(ElementTypes3D, MixedMedia) {
  const int nspec = 6;
  const int ngllz = 5;
  const int nglly = 5;
  const int ngllx = 5;

  Kokkos::View<specfem::element::medium_tag *,
               Kokkos::DefaultHostExecutionSpace>
      medium_tags("medium_tags", nspec);
  for (int i = 0; i < 3; i++) {
    medium_tags(i) = specfem::element::medium_tag::elastic;
  }
  for (int i = 3; i < 6; i++) {
    medium_tags(i) = specfem::element::medium_tag::acoustic;
  }

  Kokkos::View<specfem::element::property_tag *,
               Kokkos::DefaultHostExecutionSpace>
      property_tags("property_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    property_tags(i) = specfem::element::property_tag::isotropic;
  }

  Kokkos::View<specfem::element::attenuation_tag *,
               Kokkos::DefaultHostExecutionSpace>
      attenuation_tags("attenuation_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    attenuation_tags(i) = specfem::element::attenuation_tag::none;
  }

  Kokkos::View<specfem::element::boundary_tag *,
               Kokkos::DefaultHostExecutionSpace>
      boundary_tags("boundary_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    boundary_tags(i) = specfem::element::boundary_tag::none;
  }

  ElementTypes3D element_types(nspec, ngllz, nglly, ngllx, medium_tags,
                               property_tags, attenuation_tags, boundary_tags);

  // Test number of elements
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::elastic),
            3);
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::acoustic),
            3);

  // Test host views
  auto elastic_host =
      element_types.get_elements_on_host(specfem::element::medium_tag::elastic);
  EXPECT_EQ(elastic_host.extent(0), 3);
  EXPECT_EQ(elastic_host(0), 0);
  EXPECT_EQ(elastic_host(1), 1);
  EXPECT_EQ(elastic_host(2), 2);

  auto acoustic_host = element_types.get_elements_on_host(
      specfem::element::medium_tag::acoustic);
  EXPECT_EQ(acoustic_host.extent(0), 3);
  EXPECT_EQ(acoustic_host(0), 3);
  EXPECT_EQ(acoustic_host(1), 4);
  EXPECT_EQ(acoustic_host(2), 5);

  // Test device views
  auto acoustic_device = element_types.get_elements_on_device(
      specfem::element::medium_tag::acoustic);
  EXPECT_EQ(acoustic_device.extent(0), 3);
}

// Test 3: Medium + property combinations (isotropic only for 3D)
TEST(ElementTypes3D, MediumAndProperty) {
  const int nspec = 6;
  const int ngllz = 5;
  const int nglly = 5;
  const int ngllx = 5;

  Kokkos::View<specfem::element::medium_tag *,
               Kokkos::DefaultHostExecutionSpace>
      medium_tags("medium_tags", nspec);
  for (int i = 0; i < 3; i++) {
    medium_tags(i) = specfem::element::medium_tag::elastic;
  }
  for (int i = 3; i < 6; i++) {
    medium_tags(i) = specfem::element::medium_tag::acoustic;
  }

  Kokkos::View<specfem::element::property_tag *,
               Kokkos::DefaultHostExecutionSpace>
      property_tags("property_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    property_tags(i) = specfem::element::property_tag::isotropic;
  }

  Kokkos::View<specfem::element::attenuation_tag *,
               Kokkos::DefaultHostExecutionSpace>
      attenuation_tags("attenuation_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    attenuation_tags(i) = specfem::element::attenuation_tag::none;
  }

  Kokkos::View<specfem::element::boundary_tag *,
               Kokkos::DefaultHostExecutionSpace>
      boundary_tags("boundary_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    boundary_tags(i) = specfem::element::boundary_tag::none;
  }

  ElementTypes3D element_types(nspec, ngllz, nglly, ngllx, medium_tags,
                               property_tags, attenuation_tags, boundary_tags);

  // Test combined queries (3D only has isotropic in medium_property_elements)
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic),
            3);
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::acoustic,
                specfem::element::property_tag::isotropic),
            3);

  auto elastic_iso_host = element_types.get_elements_on_host(
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic);
  EXPECT_EQ(elastic_iso_host.extent(0), 3);
  EXPECT_EQ(elastic_iso_host(0), 0);
  EXPECT_EQ(elastic_iso_host(1), 1);
  EXPECT_EQ(elastic_iso_host(2), 2);

  auto acoustic_iso_device = element_types.get_elements_on_device(
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic);
  EXPECT_EQ(acoustic_iso_device.extent(0), 3);
}

// Test 4: Medium + property + attenuation (isotropic only for 3D)
TEST(ElementTypes3D, MediumPropertyAttenuation) {
  const int nspec = 4;
  const int ngllz = 5;
  const int nglly = 5;
  const int ngllx = 5;

  Kokkos::View<specfem::element::medium_tag *,
               Kokkos::DefaultHostExecutionSpace>
      medium_tags("medium_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    medium_tags(i) = specfem::element::medium_tag::elastic;
  }

  Kokkos::View<specfem::element::property_tag *,
               Kokkos::DefaultHostExecutionSpace>
      property_tags("property_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    property_tags(i) = specfem::element::property_tag::isotropic;
  }

  Kokkos::View<specfem::element::attenuation_tag *,
               Kokkos::DefaultHostExecutionSpace>
      attenuation_tags("attenuation_tags", nspec);
  for (int i = 0; i < 2; i++) {
    attenuation_tags(i) = specfem::element::attenuation_tag::none;
  }
  for (int i = 2; i < 4; i++) {
    attenuation_tags(i) = specfem::element::attenuation_tag::constant_isotropic;
  }

  Kokkos::View<specfem::element::boundary_tag *,
               Kokkos::DefaultHostExecutionSpace>
      boundary_tags("boundary_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    boundary_tags(i) = specfem::element::boundary_tag::none;
  }

  ElementTypes3D element_types(nspec, ngllz, nglly, ngllx, medium_tags,
                               property_tags, attenuation_tags, boundary_tags);

  // Test 3-arg queries
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic,
                specfem::element::attenuation_tag::none),
            2);
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic,
                specfem::element::attenuation_tag::constant_isotropic),
            2);

  auto none_atten_host = element_types.get_elements_on_host(
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic,
      specfem::element::attenuation_tag::none);
  EXPECT_EQ(none_atten_host.extent(0), 2);
  EXPECT_EQ(none_atten_host(0), 0);
  EXPECT_EQ(none_atten_host(1), 1);

  auto const_iso_device = element_types.get_elements_on_device(
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic,
      specfem::element::attenuation_tag::constant_isotropic);
  EXPECT_EQ(const_iso_device.extent(0), 2);
}

// Test 5: Medium + property + boundary (isotropic + none only for 3D)
TEST(ElementTypes3D, MediumPropertyBoundary) {
  const int nspec = 4;
  const int ngllz = 5;
  const int nglly = 5;
  const int ngllx = 5;

  Kokkos::View<specfem::element::medium_tag *,
               Kokkos::DefaultHostExecutionSpace>
      medium_tags("medium_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    medium_tags(i) = specfem::element::medium_tag::elastic;
  }

  Kokkos::View<specfem::element::property_tag *,
               Kokkos::DefaultHostExecutionSpace>
      property_tags("property_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    property_tags(i) = specfem::element::property_tag::isotropic;
  }

  Kokkos::View<specfem::element::attenuation_tag *,
               Kokkos::DefaultHostExecutionSpace>
      attenuation_tags("attenuation_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    attenuation_tags(i) = specfem::element::attenuation_tag::none;
  }

  Kokkos::View<specfem::element::boundary_tag *,
               Kokkos::DefaultHostExecutionSpace>
      boundary_tags("boundary_tags", nspec);
  for (int i = 0; i < nspec; i++) {
    boundary_tags(i) = specfem::element::boundary_tag::none;
  }

  ElementTypes3D element_types(nspec, ngllz, nglly, ngllx, medium_tags,
                               property_tags, attenuation_tags, boundary_tags);

  // Test boundary query (3D only supports NONE boundary)
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic,
                specfem::element::boundary_tag::none),
            4);

  auto none_boundary_host = element_types.get_elements_on_host(
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic,
      specfem::element::boundary_tag::none);
  EXPECT_EQ(none_boundary_host.extent(0), 4);
  EXPECT_EQ(none_boundary_host(0), 0);
  EXPECT_EQ(none_boundary_host(1), 1);
  EXPECT_EQ(none_boundary_host(2), 2);
  EXPECT_EQ(none_boundary_host(3), 3);

  auto none_boundary_device = element_types.get_elements_on_device(
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic,
      specfem::element::boundary_tag::none);
  EXPECT_EQ(none_boundary_device.extent(0), 4);
}

// Test 6: Empty mesh (nspec=0)
TEST(ElementTypes3D, EmptyMesh) {
  const int nspec = 0;
  const int ngllz = 5;
  const int nglly = 5;
  const int ngllx = 5;

  Kokkos::View<specfem::element::medium_tag *,
               Kokkos::DefaultHostExecutionSpace>
      medium_tags("medium_tags", nspec);
  Kokkos::View<specfem::element::property_tag *,
               Kokkos::DefaultHostExecutionSpace>
      property_tags("property_tags", nspec);
  Kokkos::View<specfem::element::attenuation_tag *,
               Kokkos::DefaultHostExecutionSpace>
      attenuation_tags("attenuation_tags", nspec);
  Kokkos::View<specfem::element::boundary_tag *,
               Kokkos::DefaultHostExecutionSpace>
      boundary_tags("boundary_tags", nspec);

  ElementTypes3D element_types(nspec, ngllz, nglly, ngllx, medium_tags,
                               property_tags, attenuation_tags, boundary_tags);

  // All queries should return 0
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::elastic),
            0);
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::acoustic),
            0);
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic),
            0);
  EXPECT_EQ(element_types.get_number_of_elements(
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic,
                specfem::element::attenuation_tag::none),
            0);

  // Host/device views should be empty
  auto host_view =
      element_types.get_elements_on_host(specfem::element::medium_tag::elastic);
  EXPECT_EQ(host_view.extent(0), 0);

  auto device_view = element_types.get_elements_on_device(
      specfem::element::medium_tag::acoustic);
  EXPECT_EQ(device_view.extent(0), 0);
}

} // namespace
