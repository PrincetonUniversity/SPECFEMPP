#include "compute/sources/sources.hpp"
#include "../test_fixture/test_fixture.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "point/sources.hpp"
#include "gtest/gtest.h"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
void check_store(specfem::compute::assembly &assembly) {

  specfem::compute::sources &sources = assembly.sources;
  const int ngllz = assembly.mesh.ngllz;
  const int ngllx = assembly.mesh.ngllx;

  const auto elements =
      assembly.sources.get_elements_on_device(MediumTag, WavefieldType);

  const int nelements = elements.size();

  constexpr int num_components =
      specfem::element::attributes<Dimension, MediumTag>::components();

  if (nelements == 0) {
    return;
  }

  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store", nelements);

  for (int i = 0; i < nelements; i++) {
    values_to_store(i) = 1.0 + i;
  }

  using PointType = specfem::point::source<Dimension, MediumTag, WavefieldType>;

  Kokkos::parallel_for(
      "check_store_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3> >(
          { 0, 0, 0 }, { nelements, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        const int ielement = elements(i);

        const auto index =
            specfem::point::index<Dimension, false>(ielement, iz, ix);
        specfem::datatype::ScalarPointViewType<type_real, num_components, false>
            stf;
        specfem::datatype::ScalarPointViewType<type_real, num_components, false>
            lagrange_interpolant;
        for (int ic = 0; ic < num_components; ic++) {
          stf(ic) = 1.0;
          lagrange_interpolant(ic) = 1.0;
        }
        PointType point(stf, lagrange_interpolant);
        specfem::compute::store_on_device(index, point, sources);
      });

  Kokkos::fence();
}

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
void check_load(specfem::compute::assembly &assembly) {

  specfem::compute::sources &sources = assembly.sources;
  const int ngllz = assembly.mesh.ngllz;
  const int ngllx = assembly.mesh.ngllx;

  const auto elements =
      sources.get_elements_on_device(MediumTag, WavefieldType);

  const int nelements = elements.size();

  constexpr int num_components =
      specfem::element::attributes<Dimension, MediumTag>::components();

  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store", nelements);

  auto h_values_to_store = Kokkos::create_mirror_view(values_to_store);

  for (int i = 0; i < nelements; i++) {
    h_values_to_store(i) = 1.0 + i;
  }

  Kokkos::deep_copy(values_to_store, h_values_to_store);

  using PointType = specfem::point::source<Dimension, MediumTag, WavefieldType>;

  Kokkos::View<PointType ***, Kokkos::DefaultExecutionSpace> point_sources(
      "point_sources", ngllz, ngllx, nelements);

  auto h_point_sources = Kokkos::create_mirror_view(point_sources);

  Kokkos::parallel_for(
      "check_load_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3> >(
          { 0, 0, 0 }, { nelements, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        const int ielement = elements(i);

        const auto index =
            specfem::point::index<Dimension, false>(ielement, iz, ix);

        PointType point;
        specfem::compute::load_on_device(index, sources, point);

        point_sources(iz, ix, i) = point;
      });

  Kokkos::fence();
  Kokkos::deep_copy(h_point_sources, point_sources);

  for (int i = 0; i < nelements; i++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        const auto &point_kernel = h_point_sources(iz, ix, i);
        for (int ic = 0; ic < num_components; ic++) {
          const auto stf = point_kernel.stf(ic);
          const auto expected = values_to_store(i);
          if (expected != stf) {
            std::ostringstream message;
            message << "Error in source computation: \n"
                    << "  ispec = " << i << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << stf << "\n"
                    << "  expected = " << expected;
            throw std::runtime_error(message.str());
          }

          const auto lagrange_interpolant =
              point_kernel.lagrange_interpolant(ic);
          if (expected != lagrange_interpolant) {
            std::ostringstream message;
            message << "Error in source computation: \n"
                    << "  ispec = " << i << "\n"
                    << "  iz = " << iz << "\n"
                    << "  ix = " << ix << "\n"
                    << "  component = " << ic << "\n"
                    << "  computed = " << lagrange_interpolant << "\n"
                    << "  expected = " << expected;
            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }
}

void test_sources(specfem::compute::assembly &assembly) {

  check_store<specfem::dimension::type::dim2,
              specfem::element::medium_tag::elastic,
              specfem::wavefield::simulation_field::forward>(assembly);

  check_load<specfem::dimension::type::dim2,
             specfem::element::medium_tag::elastic,
             specfem::wavefield::simulation_field::forward>(assembly);

  check_store<specfem::dimension::type::dim2,
              specfem::element::medium_tag::acoustic,
              specfem::wavefield::simulation_field::forward>(assembly);

  check_load<specfem::dimension::type::dim2,
             specfem::element::medium_tag::acoustic,
             specfem::wavefield::simulation_field::forward>(assembly);
}

TEST_F(ASSEMBLY, sources) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    auto assembly = std::get<1>(parameters);

    try {
      test_sources(assembly);

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
