#include "algorithms/locate_point.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "receiver/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"

void specfem::receivers::receiver::compute_receiver_array(
    const specfem::compute::mesh &mesh,
    // const specfem::compute::properties &properties,
    specfem::kokkos::HostView3d<type_real> receiver_array) {

  specfem::point::global_coordinates<specfem::dimension::type::dim2> gcoord = {
    this->x, this->z
  };
  specfem::point::local_coordinates<specfem::dimension::type::dim2> lcoord =
      specfem::algorithms::locate_point(gcoord, mesh);

  const auto xi = mesh.quadratures.gll.h_xi;
  const auto gamma = mesh.quadratures.gll.h_xi;
  const auto N = mesh.quadratures.gll.N;

  auto [hxi_receiver, hpxi_receiver] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.xi, N, xi);
  auto [hgamma_receiver, hpgamma_receiver] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.gamma, N, gamma);

  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      type_real hlagrange = hxi_receiver(ix) * hgamma_receiver(iz);

      receiver_array(0, iz, ix) = hlagrange;
      receiver_array(1, iz, ix) = hlagrange;
    }
  }

  return;
}

std::string specfem::receivers::receiver::print() const {
  std::ostringstream message;
  message << " - Receiver:\n"
          << "      Station Name = " << this->station_name << "\n"
          << "      Network Name = " << this->network_name << "\n"
          << "      Receiver Location: \n"
          << "        x = " << type_real(this->x) << "\n"
          << "        z = " << type_real(this->z) << "\n";

  return message.str();
}
