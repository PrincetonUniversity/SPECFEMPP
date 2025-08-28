#include "compute_source_array_from_vector.hpp"
#include "algorithms/interface.hpp"
#include "enumerations/macros.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

void specfem::assembly::compute_source_array_impl::from_vector(
    const specfem::sources::vector_source<specfem::dimension::type::dim3>
        &vector_source,
    Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array) {

  const int ngllz = source_array.extent(1);
  const int nglly = source_array.extent(2);
  const int ngllx = source_array.extent(3);

  // Create quadrature and compute xi/gamma arrays
  specfem::quadrature::gll::gll quadrature_x(0.0, 0.0, ngllx);
  specfem::quadrature::gll::gll quadrature_y(0.0, 0.0, nglly);
  specfem::quadrature::gll::gll quadrature_z(0.0, 0.0, ngllz);
  auto xi = quadrature_x.get_hxi();
  auto eta = quadrature_y.get_hxi();
  auto gamma = quadrature_z.get_hxi();

  // Compute lagrange interpolants at the local source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          vector_source.get_local_coordinates().xi, ngllx, xi);
  auto [heta_source, hpeta_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          vector_source.get_local_coordinates().eta, nglly, eta);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          vector_source.get_local_coordinates().gamma, ngllz, gamma);

  type_real hlagrange;

  const auto force_vector = vector_source.get_force_vector();

  int ncomponents = source_array.extent(0);

  // Sanity check
  if (ncomponents != force_vector.extent(0)) {
    KOKKOS_ABORT_WITH_LOCATION(
        "source_array_components and force_vector components do not match")
  }

  // Source array computation
  for (int iz = 0; iz < ngllz; ++iz) {
    for (int iy = 0; iy < nglly; ++iy) {
      for (int ix = 0; ix < ngllx; ++ix) {
        hlagrange = hxi_source(ix) * heta_source(iy) * hgamma_source(iz);
        for (int i = 0; i < ncomponents; ++i) {
          source_array(i, iz, iy, ix) = hlagrange * force_vector(i);
        }
      }
    }
  }

  return;
}
