#include "algorithms/locate_point.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/element_types/element_types.hpp"
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

void specfem::sources::cosserat_force::compute_source_array(
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

  if (el_type != specfem::element::medium_tag::elastic_psv_t) {
    throw std::runtime_error(
        "Cosserat force only implemented for elastic psv_t medium");
  }

  if (ncomponents != 3) {
    throw std::runtime_error(
        "Source array requires 3 components for elastic psv_t medium");
  }

  // Source array computation
  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      hlagrange = hxi_source(ix) * hgamma_source(iz);

      // Elastic P-SV-T
      if (el_type != specfem::element::medium_tag::elastic_psv_t) {
        source_array(0, iz, ix) =
            f * std::sin(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;
        source_array(1, iz, ix) =
            -1.0 * f *
            std::cos(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;

        source_array(2, iz, ix) = fc * hlagrange;
      } else {
        std::ostringstream message;
        message << "Source array computation not implemented for element type: "
                << specfem::element::to_string(el_type);
        auto message_str = message.str();
        Kokkos::abort(message_str.c_str());
      }
    }
  }
}

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
