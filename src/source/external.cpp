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
// #include "utilities.cpp"
#include "yaml-cpp/yaml.h"
#include <cmath>

void specfem::sources::external::compute_source_array(
    const specfem::compute::mesh &mesh,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      this->x, this->z);
  auto lcoord = specfem::algorithms::locate_point(coord, mesh);

  const auto xi = mesh.quadratures.gll.h_xi;
  const auto gamma = mesh.quadratures.gll.h_xi;
  const auto N = mesh.quadratures.gll.N;

  const auto el_type = element_types.get_medium_tag(lcoord.ispec);
  const int ncomponents = source_array.extent(0);

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

      if (el_type == specfem::element::medium_tag::acoustic) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "Force source requires 1 component for acoustic medium");
        }
        source_array(0, iz, ix) = hlagrange;
      } else if ((el_type == specfem::element::medium_tag::elastic_sv) ||
                 (el_type == specfem::element::medium_tag::poroelastic)) {
        if (ncomponents != 2) {
          throw std::runtime_error(
              "Force source requires 2 components for elastic medium");
        }
        if (specfem::globals::simulation_wave == specfem::wave::sh) {
          source_array(0, iz, ix) = hlagrange;
          source_array(1, iz, ix) = 0;
        } else {
          source_array(0, iz, ix) = hlagrange;
          source_array(1, iz, ix) = hlagrange;
        }
      }
    }
  }
}

std::string specfem::sources::external::print() const {

  std::ostringstream message;
  message << "- External Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}
