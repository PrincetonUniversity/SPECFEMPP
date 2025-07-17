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

template <>
void specfem::assembly::compute_source_array<specfem::dimension::type::dim2>(
    const specfem::sources::vector_source &vector_source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    specfem::kokkos::HostView3d<type_real> source_array) {

  // Not getting around the mesh input
  auto xi = mesh.h_xi;
  auto gamma = mesh.h_xi;
  const int ngllx = mesh.ngllx;
  const int ngllz = mesh.ngllz;

  // Compute lagrange interpolants at the local source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          vector_source.get_xi(), ngllx, xi);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          vector_source.get_gamma(), ngllz, gamma);

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
    for (int ix = 0; ix < ngllx; ++ix) {
      hlagrange = hxi_source(ix) * hgamma_source(iz);
      for (int i = 0; i < ncomponents; ++i) {
        source_array(i, iz, ix) = hlagrange * force_vector(i);
      }
    }
  }

  return;
}
