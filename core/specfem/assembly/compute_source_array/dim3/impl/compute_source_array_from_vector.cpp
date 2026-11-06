#include "compute_source_array_from_vector.hpp"

#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>

// Helper function to add contribution of a vector source to
// the source array using Lagrange interpolation. Also used for
// monopole contribution in Cosserat moment tensor sources.
void specfem::assembly::compute_source_array_impl::
    accumulate_vector_contribution(
        const specfem::point::local_coordinates<
            specfem::element::dimension_tag::dim3> &local_coordinates,
        const Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
            &vector,
        Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
            source_array) {

  const int ngllz = source_array.extent(1);
  const int nglly = source_array.extent(2);
  const int ngllx = source_array.extent(3);

  // Create quadrature and compute xi/eta/gamma arrays
  specfem::quadrature::gll::gll quadrature_x(0.0, 0.0, ngllx);
  specfem::quadrature::gll::gll quadrature_y(0.0, 0.0, nglly);
  specfem::quadrature::gll::gll quadrature_z(0.0, 0.0, ngllz);
  auto xi = quadrature_x.get_hxi();
  auto eta = quadrature_y.get_hxi();
  auto gamma = quadrature_z.get_hxi();

  // Compute lagrange interpolants at the local source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          local_coordinates.xi, ngllx, xi);
  auto [heta_source, hpeta_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          local_coordinates.eta, nglly, eta);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          local_coordinates.gamma, ngllz, gamma);

  type_real hlagrange;

  const int ncomponents = source_array.extent(0);

  // Sanity check
  if (ncomponents != static_cast<int>(vector.extent(0))) {
    KOKKOS_ABORT_WITH_LOCATION(
        "source_array components and vector components do not match")
  }

  // Accumulate the interpolated vector contribution onto the source array
  for (int iz = 0; iz < ngllz; ++iz) {
    for (int iy = 0; iy < nglly; ++iy) {
      for (int ix = 0; ix < ngllx; ++ix) {
        hlagrange = hxi_source(ix) * heta_source(iy) * hgamma_source(iz);
        for (int i = 0; i < ncomponents; ++i) {
          source_array(i, iz, iy, ix) += hlagrange * vector(i);
        }
      }
    }
  }

  return;
}

void specfem::assembly::compute_source_array_impl::from_vector(
    const specfem::sources::vector_source<specfem::element::dimension_tag::dim3>
        &vector_source,
    Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array) {

  // Overwrite semantics: zero the array, then accumulate the force vector.
  Kokkos::deep_copy(source_array, 0);

  specfem::assembly::compute_source_array_impl::accumulate_vector_contribution(
      vector_source.get_local_coordinates(), vector_source.get_force_vector(),
      source_array);

  return;
}
