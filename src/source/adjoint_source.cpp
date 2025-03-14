#include "source/adjoint_source.hpp"
#include "algorithms/locate_point.hpp"
#include "globals.h"

void specfem::sources::adjoint_source::compute_source_array(
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
              "Adjoint source requires 1 component for acoustic medium");
        }
        source_array(0, iz, ix) = hlagrange;
      }
      // Elastic SH
      else if (el_type == specfem::element::medium_tag::elastic_sh) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "Adjoint source requires 1 component for elastic SH medium");
        }
        source_array(0, iz, ix) = hlagrange;
      }
      // Elastic P-SV, Poroelastic, or Electromagnetic P-SV
      else if ((el_type == specfem::element::medium_tag::elastic_sv) ||
               (el_type == specfem::element::medium_tag::poroelastic) ||
               (el_type == specfem::element::medium_tag::electromagnetic_sv)) {
        if (ncomponents != 2) {
          throw std::runtime_error(
              "Adjoint source for elastic P-SV, poroelastic, or "
              "electromagnetic P-SV requires 2 components");
        }
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = hlagrange;
      }
      // Otherwise not implemented
      else {
        std::ostringstream message;
        message << "Source array computation not implemented for element type: "
                << specfem::element::to_string(el_type);
        auto message_str = message.str();
        Kokkos::abort(message_str.c_str());
      }
    }
  }
}

std::string specfem::sources::adjoint_source::print() const {

  std::ostringstream message;
  message << "- Adjoint Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";
  return message.str();
}
