#include "source/adjoint_source.hpp"
#include "algorithms/locate_point.hpp"
#include "globals.h"

void specfem::sources::adjoint_source::compute_source_array(
    const specfem::compute::mesh &mesh,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::properties &properties,
    specfem::kokkos::HostView3d<type_real> source_array) {

  specfem::point::gcoord2 coord = specfem::point::gcoord2(this->x, this->z);
  auto lcoord = specfem::algorithms::locate_point(coord, mesh);

  const auto xi = mesh.quadratures.gll.h_xi;
  const auto gamma = mesh.quadratures.gll.h_xi;
  const auto N = mesh.quadratures.gll.N;

  const auto el_type = properties.h_element_types(lcoord.ispec);

  // Compute lagrange interpolants at the source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.xi, N, xi);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.gamma, N, gamma);

  type_real hlagrange;

  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      hlagrange = hxi_source(ix) * hgamma_source(iz);

      if (el_type == specfem::element::medium_tag::acoustic ||
          (el_type == specfem::element::medium_tag::elastic &&
           specfem::globals::simulation_wave == specfem::wave::sh)) {
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = 0;
      } else if ((el_type == specfem::element::medium_tag::elastic &&
                  specfem::globals::simulation_wave == specfem::wave::p_sv) ||
                 el_type == specfem::element::medium_tag::poroelastic) {
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = hlagrange;
      }
    }
  }
};
