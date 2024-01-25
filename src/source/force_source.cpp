#include "algorithms/locate_point.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/specfem_enums.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "point/coordinates.hpp"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities.cpp"
#include "yaml-cpp/yaml.h"
#include <cmath>

void specfem::sources::force::compute_source_array(
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

      if (el_type == specfem::enums::element::type::acoustic ||
          (el_type == specfem::enums::element::type::elastic &&
           specfem::globals::simulation_wave == specfem::wave::sh)) {
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = hlagrange;
      } else if ((el_type == specfem::enums::element::type::elastic &&
                  specfem::globals::simulation_wave == specfem::wave::p_sv) ||
                 el_type == specfem::enums::element::type::poroelastic) {
        type_real tempx = sin(angle) * hlagrange;
        source_array(0, iz, ix) = tempx;
        type_real tempz = -1.0 * cos(angle) * hlagrange;
        source_array(1, iz, ix) = tempz;
      }
    }
  }
};

std::string specfem::sources::force::print() const {

  std::ostringstream message;
  message << "- Force Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}
