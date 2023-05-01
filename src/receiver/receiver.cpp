#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "receiver/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"

void specfem::receivers::receiver::locate(
    const specfem::kokkos::HostView2d<type_real> coord,
    const specfem::kokkos::HostMirror3d<int> h_ibool,
    const specfem::kokkos::HostMirror1d<type_real> xigll,
    const specfem::kokkos::HostMirror1d<type_real> zigll, const int nproc,
    const specfem::kokkos::HostView2d<type_real> coorg,
    const specfem::kokkos::HostView2d<int> knods, const int npgeo,
    const specfem::kokkos::HostMirror1d<specfem::elements::type> ispec_type,
    const specfem::MPI::MPI *mpi) {
  std::tie(this->xi, this->gamma, this->ispec, this->islice) =
      specfem::utilities::locate(coord, h_ibool, xigll, zigll, nproc, this->x,
                                 this->z, coorg, knods, npgeo, mpi);
  if (this->islice == mpi->get_rank()) {
    this->el_type = ispec_type(ispec);
  }
}

void specfem::receivers::receiver::check_locations(
    const type_real xmin, const type_real xmax, const type_real zmin,
    const type_real zmax, const specfem::MPI::MPI *mpi) {
  specfem::utilities::check_locations(this->x, this->z, xmin, xmax, zmin, zmax,
                                      mpi);
}

void specfem::receivers::receiver::compute_receiver_array(
    const specfem::quadrature::quadrature *quadx,
    const specfem::quadrature::quadrature *quadz,
    specfem::kokkos::HostView3d<type_real> receiver_array) {
  type_real xi = this->xi;
  type_real gamma = this->gamma;

  auto [hxir, hpxir] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          xi, quadx->get_N(), quadx->get_hxi());
  auto [hgammar, hpgammar] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          gamma, quadz->get_N(), quadz->get_hxi());

  int nquadx = quadx->get_N();
  int nquadz = quadz->get_N();

  type_real hlagrange;

  for (int i = 0; i < nquadx; i++) {
    for (int j = 0; j < nquadz; j++) {
      hlagrange = hxir(i) * hgammar(j);
      receiver_array(j, i, 0) = hlagrange;
      receiver_array(j, i, 1) = hlagrange;
    }
  }
}

std::string specfem::receivers::receiver::print() const {
  std::ostringstream message;
  message << " - Receiver:\n"
          << "      Station Name = " << this->station_name << "\n"
          << "      Network Name = " << this->network_name << "\n"
          << "      Receiver Location: \n"
          << "        x = " << type_real(this->x) << "\n"
          << "        z = " << type_real(this->z) << "\n"
          << "        xi = " << this->xi << "\n"
          << "        gamma = " << this->gamma << "\n"
          << "        ispec = " << this->ispec << "\n"
          << "        islice = " << this->islice << "\n";

  return message.str();
}
