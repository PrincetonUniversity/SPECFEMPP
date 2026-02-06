
// Check if the wavefield is correctly generated for the given element.
// Given an element index, we check if the wavefield component was correctly
// generated. We compare the generated wavefield with the expected wavefield.

// Expected wavefield:
// - When a component is present as on of the primary components for a medium
// type.
//      We just check if the wavefield component is equal to 1.0.

// - When a component is not present as on of the primary components for a
// medium type.
//      We check that the value is equal to 0.0.
//      Since, the computed strain for a uniform wavefield is zero.

#pragma once
#include "specfem/enums.hpp"
#include "specfem/point.hpp"

template <specfem::enums::wavefield component,
          specfem::element::medium_tag medium,
          specfem::element::property_tag property>
class test_helper {

public:
  test_helper(const int ispec,
              const Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                 Kokkos::HostSpace> &wavefield,
              specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                  &assembly)
      : ispec(ispec), wavefield(wavefield), assembly(assembly) {}

  void test() {

    constexpr static int num_components =
        specfem::element::attributes<specfem::element::dimension_tag::dim2,
                                     medium>::components;

    const int ngllz = assembly.mesh.element_grid.ngllz;
    const int ngllx = assembly.mesh.element_grid.ngllx;

    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {

        for (int ic = 0; ic < num_components; ic++) {
          const auto computed = wavefield(ispec, iz, ix, ic);
          const auto expected = 1.0;

          if (std::abs(computed - expected) > 1.0e-4) {
            std::ostringstream message;
            message << "Error in elastic wavefield computation: \n"
                    << "  ispec = " << ispec << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << computed << "\n"
                    << "  expected = " << expected;
            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }

private:
  const int ispec;
  const Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
      &wavefield;
  specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly;
};

template <specfem::element::medium_tag medium,
          specfem::element::property_tag property>
class test_helper<specfem::enums::wavefield::pressure, medium, property> {
public:
  test_helper(const int ispec,
              const Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                 Kokkos::HostSpace> &wavefield,
              specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                  &assembly)
      : ispec(ispec), wavefield(wavefield), assembly(assembly) {}

  void test() {

    constexpr static int num_components = specfem::element::attributes<
        specfem::element::dimension_tag::dim2,
        specfem::element::medium_tag::acoustic>::components;

    const int ngllz = assembly.mesh.element_grid.ngllz;
    const int ngllx = assembly.mesh.element_grid.ngllx;

    using PointProperties =
        specfem::point::properties<specfem::element::dimension_tag::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   specfem::element::property_tag::isotropic,
                                   false>;

    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {

        const specfem::point::index<specfem::element::dimension_tag::dim2,
                                    false>
            index(ispec, iz, ix);

        PointProperties point_properties;
        specfem::assembly::load_on_host(index, assembly.properties,
                                        point_properties);

        for (int ic = 0; ic < num_components; ic++) {
          const auto computed =
              wavefield(ispec, iz, ix, ic) /
              ((point_properties.lambdaplus2mu() + point_properties.lambda()) /
               2.0);
          const auto expected = 0.0;

          if (std::abs(computed - expected) > 1.0e-4) {
            std::ostringstream message;
            message << "Error in elastic wavefield computation: \n"
                    << "  ispec = " << ispec << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << computed << "\n"
                    << "  expected = " << expected;
            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }

private:
  const int ispec;
  const Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
      &wavefield;
  specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly;
};

template <specfem::enums::wavefield component>
class test_helper<component, specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic> {

public:
  test_helper(const int ispec,
              const Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                 Kokkos::HostSpace> &wavefield,
              specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                  &assembly)
      : ispec(ispec), wavefield(wavefield), assembly(assembly) {}

  void test() {

    constexpr static int num_components = specfem::element::attributes<
        specfem::element::dimension_tag::dim2,
        specfem::element::medium_tag::acoustic>::components;

    const int ngllz = assembly.mesh.element_grid.ngllz;
    const int ngllx = assembly.mesh.element_grid.ngllx;

    using PointProperties =
        specfem::point::properties<specfem::element::dimension_tag::dim2,
                                   specfem::element::medium_tag::acoustic,
                                   specfem::element::property_tag::isotropic,
                                   false>;

    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {

        const specfem::point::index<specfem::element::dimension_tag::dim2,
                                    false>
            index(ispec, iz, ix);

        PointProperties point_properties;
        specfem::assembly::load_on_host(index, assembly.properties,
                                        point_properties);

        for (int ic = 0; ic < num_components; ic++) {
          const auto computed =
              wavefield(ispec, iz, ix, ic) / point_properties.rho_inverse();
          const auto expected = 0.0;

          if (std::abs(computed - expected) > 1.0e-4) {
            std::ostringstream message;
            message << "Error in acoustic wavefield computation: \n"
                    << "  ispec = " << ispec << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << computed << "\n"
                    << "  expected = " << expected;
            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }

private:
  const int ispec;
  const Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
      &wavefield;
  specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly;
};

template <>
class test_helper<specfem::enums::wavefield::pressure,
                  specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic> {

public:
  test_helper(const int ispec,
              const Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                 Kokkos::HostSpace> &wavefield,
              specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                  &assembly)
      : ispec(ispec), wavefield(wavefield), assembly(assembly) {}

  void test() {

    constexpr static int num_components = specfem::element::attributes<
        specfem::element::dimension_tag::dim2,
        specfem::element::medium_tag::acoustic>::components;

    const int ngllz = assembly.mesh.element_grid.ngllz;
    const int ngllx = assembly.mesh.element_grid.ngllx;

    using PointProperties =
        specfem::point::properties<specfem::element::dimension_tag::dim2,
                                   specfem::element::medium_tag::acoustic,
                                   specfem::element::property_tag::isotropic,
                                   false>;

    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {

        const specfem::point::index<specfem::element::dimension_tag::dim2,
                                    false>
            index(ispec, iz, ix);

        PointProperties point_properties;
        specfem::assembly::load_on_host(index, assembly.properties,
                                        point_properties);

        for (int ic = 0; ic < num_components; ic++) {
          const auto computed = wavefield(ispec, iz, ix, ic);
          const auto expected = -1.0;

          if (std::abs(computed - expected) > 1.0e-4) {
            std::ostringstream message;
            message << "Error in acoustic wavefield computation: \n"
                    << "  ispec = " << ispec << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << computed << "\n"
                    << "  expected = " << expected;
            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }

private:
  const int ispec;
  const Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
      &wavefield;
  specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly;
};
