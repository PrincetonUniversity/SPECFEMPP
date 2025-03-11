#include "IO/interface.hpp"
#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include "algorithms/interpolate.hpp"
#include "algorithms/locate_point.hpp"
#include "compute/compute_mesh.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "quadrature/gll/gll.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

inline type_real function1(const type_real x, const type_real z) {
  return std::sqrt(x * x + z * z);
}

TEST(ALGORITHMS, interpolate_function) {

  std::string database_file = "algorithms/serial/database.bin";

  // Read Mesh database
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();
  specfem::mesh::mesh mesh = specfem::IO::read_2d_mesh(database_file, mpi);

  constexpr int N = 5;

  // Quadratures
  specfem::quadrature::gll::gll gll(0.0, 0.0, N);
  specfem::quadrature::quadratures quadratures(gll);

  // Assemble
  specfem::compute::mesh assembly(mesh.tags, mesh.control_nodes, quadratures);

  const auto xi = assembly.quadratures.gll.h_xi;
  const auto gamma = assembly.quadratures.gll.h_xi;

  const type_real xi_target = 0.15;
  const type_real gamma_target = 0.15;
  const int ispec_target = 1452;

  specfem::kokkos::HostView2d<type_real> polynomial("polynomial", N, N);
  specfem::kokkos::HostView2d<type_real> function("function", N, N);

  auto [hxi, hpxi] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          xi_target, N, xi);
  auto [hgamma, hpgamma] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          gamma_target, N, gamma);

  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      polynomial(iz, ix) = hxi(ix) * hgamma(iz);
      specfem::point::local_coordinates<specfem::dimension::type::dim2>
          lcoord = { ispec_target, gamma(iz), xi(ix) };
      auto gcoord = specfem::algorithms::locate_point(lcoord, assembly);

      function(iz, ix) = function1(gcoord.x, gcoord.z);
    }
  }

  auto function_interpolated =
      specfem::algorithms::interpolate_function(polynomial, function);

  const specfem::point::local_coordinates<specfem::dimension::type::dim2>
      lcoord = { ispec_target, gamma_target, xi_target };
  auto gcoord = specfem::algorithms::locate_point(lcoord, assembly);

  type_real function_value = function1(gcoord.x, gcoord.z);

  EXPECT_NEAR(function_value, function_interpolated, 1e-3);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
