#include "specfem/assembly/compute_source_array/dim2/impl/compute_source_array_from_vector.hpp"
#include "../../test_fixture/test_fixture.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "test_macros.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

// Helper function to test a vector source
template <typename SourceType>
void test_vector_source(const std::string &source_name, SourceType &source,
                        int ngll) {
  SCOPED_TRACE("Testing " + source_name);

  // Create quadrature to get GLL points
  specfem::quadrature::gll::gll quadrature(0.0, 0.0, ngll);
  auto xi_gamma_points = quadrature.get_hxi();

  // Get the force vector for this source to determine number of components
  auto force_vector = source.get_force_vector();
  int ncomponents = force_vector.extent(0);

  // Create source array for testing
  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll);

  // Loop over all GLL points
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      SCOPED_TRACE("Testing GLL point (ix=" + std::to_string(ix) +
                   ", iz=" + std::to_string(iz) + ")");

      // Set source location to this GLL point
      specfem::point::local_coordinates<specfem::dimension::type::dim2>
          local_coords(0, xi_gamma_points(ix), xi_gamma_points(iz));
      source.set_local_coordinates(local_coords);

      // Initialize source array to zero
      for (int ic = 0; ic < ncomponents; ++ic) {
        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            source_array(ic, jz, jx) = 0.0;
          }
        }
      }

      // Compute source array using the implementation function
      specfem::assembly::compute_source_array_impl::from_vector(source,
                                                                source_array);

      // The source array should have non-zero values only at the GLL point
      // where the source is located, and those values should equal the force
      // vector
      for (int ic = 0; ic < ncomponents; ++ic) {
        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            if (jx == ix && jz == iz) {
              // At the source location, value should equal force vector
              // component
              EXPECT_NEAR(source_array(ic, jz, jx), force_vector(ic), 1e-12)
                  << "Component " << ic << " at source location (" << jx << ","
                  << jz << ") should equal force vector component";
            } else {
              // Away from source location, value should be zero (within
              // tolerance)
              EXPECT_NEAR(source_array(ic, jz, jx), 0.0, 1e-12)
                  << "Component " << ic << " at location (" << jx << "," << jz
                  << ") should be zero when source is at (" << ix << "," << iz
                  << ")";
            }
          }
        }
      }
    }
  }
}

TEST(ASSEMBLY_NO_LOAD, compute_source_array_from_vector) {

  const int ngll = 5;

  // Test Force sources with different component combinations

  // (1,0) - force in x direction only (angle=0)
  {
    specfem::sources::force<specfem::dimension::type::dim2> force_x(
        0.0, 0.0, 0.0, // x, z, angle (angle=0 means force in x direction)
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    force_x.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_vector_source("Force X-direction (1,0)", force_x, ngll);
  }

  // (0,1) - force in z direction only (angle=90)
  {
    specfem::sources::force<specfem::dimension::type::dim2> force_z(
        0.0, 0.0, 90.0, // angle=90 means force in z direction
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    force_z.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_vector_source("Force Z-direction (0,1)", force_z, ngll);
  }

  // (1,1) - force in both directions (angle=45)
  {
    specfem::sources::force<specfem::dimension::type::dim2> force_both(
        0.0, 0.0, 45.0, // angle=45 means equal force in both directions
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    force_both.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_vector_source("Force Both directions (1,1)", force_both, ngll);
  }

  // (0,0) - zero force (amplitude=0)
  {
    specfem::sources::force<specfem::dimension::type::dim2> force_zero(
        0.0, 0.0, 0.0, // angle doesn't matter
        std::make_unique<specfem::forcing_function::Ricker>(
            10, 0.01, 0.0, 0.0, 1.0, false), // amplitude = 0
        specfem::wavefield::simulation_field::forward);
    force_zero.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_vector_source("Force Zero (0,0)", force_zero, ngll);
  }

  // Test Adjoint sources (2 components)
  {
    specfem::sources::adjoint_source<specfem::dimension::type::dim2> adjoint(
        0.0, 0.0, // x, z
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        "STA1", "NET1");
    adjoint.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_vector_source("Adjoint Source", adjoint, ngll);
  }

  // Test Cosserat force sources (3 components: vx, vz, rotation)

  // (1,0,0) - force in x direction only
  {
    specfem::sources::cosserat_force<specfem::dimension::type::dim2> cosserat_x(
        0.0, 0.0, // x, z
        1.0, 0.0, // f=1 (elastic force), fc=0 (no rotational force)
        0.0,      // angle=0 means force in x direction
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    cosserat_x.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
    test_vector_source("Cosserat X-direction (1,0,0)", cosserat_x, ngll);
  }

  // (0,1,0) - force in z direction only
  {
    specfem::sources::cosserat_force<specfem::dimension::type::dim2> cosserat_z(
        0.0, 0.0, // x, z
        1.0, 0.0, // f=1 (elastic force), fc=0 (no rotational force)
        90.0,     // angle=90 means force in z direction
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    cosserat_z.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
    test_vector_source("Cosserat Z-direction (0,1,0)", cosserat_z, ngll);
  }

  // (0,0,1) - rotational force only
  {
    specfem::sources::cosserat_force<specfem::dimension::type::dim2>
        cosserat_rot(0.0, 0.0, // x, z
                     0.0,
                     1.0, // f=0 (no elastic force), fc=1 (rotational force)
                     0.0, // angle doesn't matter for pure rotation
                     std::make_unique<specfem::forcing_function::Ricker>(
                         10, 0.01, 1.0, 0.0, 1.0, false),
                     specfem::wavefield::simulation_field::forward);
    cosserat_rot.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
    test_vector_source("Cosserat Rotation (0,0,1)", cosserat_rot, ngll);
  }

  // (1,1,1) - force in all directions
  {
    specfem::sources::cosserat_force<specfem::dimension::type::dim2>
        cosserat_all(0.0, 0.0, // x, z
                     1.0, 1.0, // f=1 (elastic force), fc=1 (rotational force)
                     45.0,     // angle=45 means equal force in both directions
                     std::make_unique<specfem::forcing_function::Ricker>(
                         10, 0.01, 1.0, 0.0, 1.0, false),
                     specfem::wavefield::simulation_field::forward);
    cosserat_all.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
    test_vector_source("Cosserat All directions (1,1,1)", cosserat_all, ngll);
  }

  // Test External source (2 components)
  {
    specfem::sources::external<specfem::dimension::type::dim2> external(
        0.0, 0.0, // x, z
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    external.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_vector_source("External Source", external, ngll);
  }
}
