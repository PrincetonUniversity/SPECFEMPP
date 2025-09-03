#include "specfem/assembly/compute_source_array/dim3/impl/compute_source_array_from_vector.hpp"
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

// Helper function to test a 3D vector source
template <typename SourceType>
void test_vector_source_3d(const std::string &source_name, SourceType &source,
                           int ngll) {
  SCOPED_TRACE("Testing " + source_name);

  // Create quadrature to get GLL points
  specfem::quadrature::gll::gll quadrature(0.0, 0.0, ngll);
  auto xi_eta_gamma_points = quadrature.get_hxi();

  // Get the force vector for this source to determine number of components
  auto force_vector = source.get_force_vector();
  int ncomponents = force_vector.extent(0);

  // Create source array for testing (4D: [components, z, y, x])
  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll, ngll);

  // Loop over all GLL points
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        SCOPED_TRACE("Testing GLL point (ix=" + std::to_string(ix) + ", iy=" +
                     std::to_string(iy) + ", iz=" + std::to_string(iz) + ")");

        // Set source location to this GLL point
        specfem::point::local_coordinates<specfem::dimension::type::dim3>
            local_coords(0, xi_eta_gamma_points(ix), xi_eta_gamma_points(iy),
                         xi_eta_gamma_points(iz));
        source.set_local_coordinates(local_coords);

        // Initialize source array to zero
        for (int ic = 0; ic < ncomponents; ++ic) {
          for (int jz = 0; jz < ngll; ++jz) {
            for (int jy = 0; jy < ngll; ++jy) {
              for (int jx = 0; jx < ngll; ++jx) {
                source_array(ic, jz, jy, jx) = 0.0;
              }
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
            for (int jy = 0; jy < ngll; ++jy) {
              for (int jx = 0; jx < ngll; ++jx) {
                if (jx == ix && jy == iy && jz == iz) {
                  // At the source location, value should equal force vector
                  // component
                  EXPECT_NEAR(source_array(ic, jz, jy, jx), force_vector(ic),
                              1e-12)
                      << "Component " << ic << " at source location (" << jx
                      << "," << jy << "," << jz
                      << ") should equal force vector component";
                } else {
                  // Away from source location, value should be zero (within
                  // tolerance)
                  EXPECT_NEAR(source_array(ic, jz, jy, jx), 0.0, 1e-12)
                      << "Component " << ic << " at location (" << jx << ","
                      << jy << "," << jz
                      << ") should be zero when source is at (" << ix << ","
                      << iy << "," << iz << ")";
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(ASSEMBLY_NO_LOAD, compute_source_array_from_vector_3d) {

  const int ngll = 5;

  // Test Force sources with different component combinations

  // (1,0,0) - force in x direction only
  {
    specfem::sources::force<specfem::dimension::type::dim3> force_x(
        0.0, 0.0, 0.0, // x, y, z
        1.0, 0.0, 0.0, // fx, fy, fz
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    force_x.set_medium_tag(specfem::element::medium_tag::elastic);
    test_vector_source_3d("Force X-direction (1,0,0)", force_x, ngll);
  }

  // (0,1,0) - force in y direction only
  {
    specfem::sources::force<specfem::dimension::type::dim3> force_y(
        0.0, 0.0, 0.0, // x, y, z
        0.0, 1.0, 0.0, // fx, fy, fz
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    force_y.set_medium_tag(specfem::element::medium_tag::elastic);
    test_vector_source_3d("Force Y-direction (0,1,0)", force_y, ngll);
  }

  // (0,0,1) - force in z direction only
  {
    specfem::sources::force<specfem::dimension::type::dim3> force_z(
        0.0, 0.0, 0.0, // x, y, z
        0.0, 0.0, 1.0, // fx, fy, fz
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    force_z.set_medium_tag(specfem::element::medium_tag::elastic);
    test_vector_source_3d("Force Z-direction (0,0,1)", force_z, ngll);
  }

  // (1,1,1) - force in all directions
  {
    specfem::sources::force<specfem::dimension::type::dim3> force_all(
        0.0, 0.0, 0.0, // x, y, z
        1.0, 1.0, 1.0, // fx, fy, fz
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    force_all.set_medium_tag(specfem::element::medium_tag::elastic);
    test_vector_source_3d("Force All directions (1,1,1)", force_all, ngll);
  }

  // (0,0,0) - zero force (amplitude=0)
  {
    specfem::sources::force<specfem::dimension::type::dim3> force_zero(
        0.0, 0.0, 0.0, // x, y, z
        0.0, 0.0, 0.0, // fx, fy, fz
        std::make_unique<specfem::forcing_function::Ricker>(
            10, 0.01, 1.0, 0.0, 1.0, false), // normal amplitude, but zero force
                                             // components
        specfem::wavefield::simulation_field::forward);
    force_zero.set_medium_tag(specfem::element::medium_tag::elastic);
    test_vector_source_3d("Force Zero (0,0,0)", force_zero, ngll);
  }

  // Test Adjoint sources (3 components)
  // {
  //   specfem::sources::adjoint_source<specfem::dimension::type::dim3> adjoint(
  //       0.0, 0.0, 0.0, // x, y, z
  //       std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0,
  //       0.0,
  //                                                           1.0, false),
  //       "STA1", "NET1");
  //   adjoint.set_medium_tag(specfem::element::medium_tag::elastic);
  //   test_vector_source_3d("Adjoint Source", adjoint, ngll);
  // }

  // Test External source (3 components)
  // {
  //   specfem::sources::external<specfem::dimension::type::dim3> external(
  //       0.0, 0.0, 0.0, // x, y, z
  //       std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0,
  //       0.0,
  //                                                           1.0, false),
  //       specfem::wavefield::simulation_field::forward);
  //   external.set_medium_tag(specfem::element::medium_tag::elastic);
  //   test_vector_source_3d("External Source", external, ngll);
  // }
}
