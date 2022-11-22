#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/reciever.h"
#include "../include/specfem_mpi.h"
#include "../include/util.h"
void specfem::recievers::reciever::locate(
    const specfem::HostView3d<int> ibool,
    const specfem::HostView2d<type_real> coord,
    const specfem::HostMirror1d<type_real> xigll,
    const specfem::HostMirror1d<type_real> zigll, const int nproc,
    const specfem::HostView3d<type_real> coorg,
    const specfem::HostView2d<int> knods, const int npgeo,
    const specfem::HostView1d<element_type> ispec_type,
    const specfem::MPI *mpi) {
  std::tie(this->xi, this->gamma, this->ispec, this->islice) =
      specfem::utilities::locate(ibool, coord, xigll, zigll, nproc, this->x,
                                 this->z, coorg, knods, npgeo, mpi);
  if (this->islice == mpi->get_rank())
    this->el_type = ispec_type(ispec);
}

void specfem::recievers::reciever::check_locations(const type_real xmin,
                                                   const type_real xmax,
                                                   const type_real zmin,
                                                   const type_real zmax,
                                                   const specfem::MPI *mpi) {
  specfem::utilities::check_locations(xmin, xmax, zmin, zmax, mpi);
}

void specfem::recievers::reciever::compute_reciever_array(
    specfem::quadrature &quadx, specfem::quadrature &quadz,
    specfem::HostView3d<type_real> reciever_array) {
  int ispec = this->ispec;
  type_real xi = this->xi;
  type_real gamma = this->gamma;
  type_real angle = this->angle;
  element_type el_type = this->el_type;
  wave_type wave = this->wave;

  auto [hxis, hpxis] = Lagrange::compute_lagrange_interpolants(
      xi, quadx.get_N(), quadx.get_hxi());
  auto [hgamma, hpgamma] = Lagrange::compute_lagrange_interpolants(
      gamma, quadz.get_N(), quadz.get_hxi());

  int nquadx = quadx.get_N();
  int nquadz = quadz.get_N();

  type_real hlagrange;

  for (int i = 0; i < nquadx; i++) {
    for (int j = 0; j < nquadz; j++) {
      hlagrange = hxis(i) * hgammas(j);
      reciever_array(j, i, 0) = hlagrange;
      reciever_array(j, i, 1) = hlagrange;
    }
  }
}
