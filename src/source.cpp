#include "../include/source.h"
#include "../include/config.h"
#include "../include/jacobian.h"
#include "../include/kokkos_abstractions.h"
#include "../include/lagrange_poly.h"
#include "../include/source_time_function.h"
#include "../include/specfem_mpi.h"
#include "../include/timescheme.h"
#include "../include/utils.h"

using LayoutStride = Kokkos::LayoutStride;

void specfem::sources::source::check_locations(const type_real xmin,
                                               const type_real xmax,
                                               const type_real zmin,
                                               const type_real zmax,
                                               const specfem::MPI::MPI *mpi) {
  specfem::utilities::check_locations(this->get_x(), this->get_z(), xmin, xmax,
                                      zmin, zmax, mpi);
}

void specfem::sources::force::locate(
    const specfem::HostView3d<int> ibool,
    const specfem::HostView2d<type_real> coord,
    const specfem::HostMirror1d<type_real> xigll,
    const specfem::HostMirror1d<type_real> zigll, const int nproc,
    const specfem::HostView2d<type_real> coorg,
    const specfem::HostView2d<int> knods, const int npgeo,
    const specfem::HostView1d<element_type> ispec_type,
    const specfem::MPI::MPI *mpi) {
  std::tie(this->xi, this->gamma, this->ispec, this->islice) =
      specfem::utilities::locate(ibool, coord, xigll, zigll, nproc,
                                 this->get_x(), this->get_z(), coorg, knods,
                                 npgeo, mpi);
  if (this->islice == mpi->get_rank()) {
    this->el_type = ispec_type(ispec);
  }
}

void specfem::sources::moment_tensor::locate(
    const specfem::HostView3d<int> ibool,
    const specfem::HostView2d<type_real> coord,
    const specfem::HostMirror1d<type_real> xigll,
    const specfem::HostMirror1d<type_real> zigll, const int nproc,
    const specfem::HostView2d<type_real> coorg,
    const specfem::HostView2d<int> knods, const int npgeo,
    const specfem::HostView1d<element_type> ispec_type,
    const specfem::MPI::MPI *mpi) {
  std::tie(this->xi, this->gamma, this->ispec, this->islice) =
      specfem::utilities::locate(ibool, coord, xigll, zigll, nproc,
                                 this->get_x(), this->get_z(), coorg, knods,
                                 npgeo, mpi);

  if (this->islice == mpi->get_rank()) {
    if (ispec_type(ispec) != elastic)
      throw std::runtime_error(
          "Found a Moment-tensor source in acoustic/poroelastic element");
  }
  int ngnod = knods.extent(0);
  this->s_coorg = specfem::HostView2d<type_real>(
      "specfem::sources::moment_tensor::s_coorg", ndim, ngnod);

  // Store s_coorg for better caching
  for (int in = 0; in < ngnod; in++) {
    this->s_coorg(0, in) = coorg(0, knods(in, ispec));
    this->s_coorg(1, in) = coorg(1, knods(in, ispec));
  }

  return;
}

void specfem::sources::force::compute_source_array(
    specfem::quadrature::quadrature &quadx,
    specfem::quadrature::quadrature &quadz,
    specfem::HostView3d<type_real> source_array) {

  type_real xi = this->xi;
  type_real gamma = this->gamma;
  type_real angle = this->angle;
  element_type el_type = this->el_type;
  wave_type wave = this->wave;

  auto [hxis, hpxis] = Lagrange::compute_lagrange_interpolants(
      xi, quadx.get_N(), quadx.get_hxi());
  auto [hgammas, hpgammas] = Lagrange::compute_lagrange_interpolants(
      gamma, quadz.get_N(), quadz.get_hxi());

  int nquadx = quadx.get_N();
  int nquadz = quadz.get_N();

  type_real hlagrange;

  for (int i = 0; i < nquadx; i++) {
    for (int j = 0; j < nquadz; j++) {
      hlagrange = hxis(i) * hgammas(j);

      if (el_type == acoustic || (el_type == elastic && wave == sh)) {
        source_array(j, i, 0) = hlagrange;
        source_array(j, i, 1) = hlagrange;
      } else if ((el_type == elastic && wave == p_sv) ||
                 el_type == poroelastic) {
        source_array(j, i, 0) = sin(angle) * hlagrange;
        source_array(j, i, 1) = -1.0 * cos(angle) * hlagrange;
      }
    }
  }
};

void specfem::sources::moment_tensor::compute_source_array(
    specfem::quadrature::quadrature &quadx,
    specfem::quadrature::quadrature &quadz,
    specfem::HostView3d<type_real> source_array) {

  type_real xi = this->xi;
  type_real gamma = this->gamma;
  type_real Mxx = this->Mxx;
  type_real Mxz = this->Mxz;
  type_real Mzz = this->Mzz;
  auto s_coorg = this->s_coorg;
  int ngnod = s_coorg.extent(1);

  auto [hxis, hpxis] = Lagrange::compute_lagrange_interpolants(
      xi, quadx.get_N(), quadx.get_hxi());
  auto [hgammas, hpgammas] = Lagrange::compute_lagrange_interpolants(
      gamma, quadz.get_N(), quadz.get_hxi());

  int nquadx = quadx.get_N();
  int nquadz = quadz.get_N();

  type_real hlagrange;
  type_real dxis_dx = 0;
  type_real dxis_dz = 0;
  type_real dgammas_dx = 0;
  type_real dgammas_dz = 0;

  for (int i = 0; i < nquadx; i++) {
    for (int j = 0; j < nquadz; j++) {
      type_real xil = quadx.get_hxi()(i);
      type_real gammal = quadz.get_hxi()(j);
      auto [xix, xiz, gammax, gammaz] =
          jacobian::compute_inverted_derivatives(s_coorg, ngnod, xil, gammal);
      hlagrange = hxis(i) * hgammas(j);
      dxis_dx += hlagrange * xix;
      dxis_dz += hlagrange * xiz;
      dgammas_dx += hlagrange * gammax;
      dgammas_dz += hlagrange * gammaz;
    }
  }

  for (int i = 0; i < nquadx; i++) {
    for (int j = 0; j < nquadz; j++) {
      type_real dsrc_dx = (hpxis(i) * dxis_dx) * hgammas(j) +
                          hxis(i) * (hpgammas(j) * dgammas_dx);
      type_real dsrc_dz = (hpxis(i) * dxis_dz) * hgammas(j) +
                          hxis(i) * (hpgammas(j) * dgammas_dz);

      source_array(j, i, 0) += Mxx * dsrc_dx + Mxz * dsrc_dz;
      source_array(j, i, 1) += Mxz * dsrc_dx + Mzz * dsrc_dz;
    }
  }
};

void specfem::sources::force::check_locations(const type_real xmin,
                                              const type_real xmax,
                                              const type_real zmin,
                                              const type_real zmax,
                                              const specfem::MPI::MPI *mpi) {

  specfem::utilities::check_locations(this->get_x(), this->get_z(), xmin, xmax,
                                      zmin, zmax, mpi);
  mpi->cout(
      "ToDo:: Need to implement a check to see if acoustic source lies on an "
      "acoustic surface");
}

specfem::sources::force::force(type_real x, type_real z, type_real angle,
                               type_real tshift, type_real f0, type_real factor,
                               std::string forcing_type, wave_type wave)
    : x(x), z(z), angle(angle), wave(wave) {

  bool use_trick_for_better_pressure = true;

  if (forcing_type == "Dirac") {
    this->forcing_function = new specfem::forcing_function::Dirac(
        f0, tshift, factor, use_trick_for_better_pressure);
  }
};

specfem::sources::force::force(specfem::utilities::force_source &force_source,
                               wave_type wave)
    : x(force_source.x), z(force_source.z), angle(force_source.angle),
      wave(wave) {

  bool use_trick_for_better_pressure = true;

  if (force_source.stf_type == "Dirac") {
    this->forcing_function = new specfem::forcing_function::Dirac(
        force_source.f0, force_source.tshift, force_source.factor,
        use_trick_for_better_pressure);
  }
};

specfem::sources::moment_tensor::moment_tensor(type_real x, type_real z,
                                               type_real Mxx, type_real Mxz,
                                               type_real Mzz, type_real tshift,
                                               type_real f0, type_real factor,
                                               std::string forcing_type)
    : x(x), z(z), Mxx(Mxx), Mxz(Mxz), Mzz(Mzz) {

  bool use_trick_for_better_pressure = true;

  if (forcing_type == "Dirac") {
    this->forcing_function = new specfem::forcing_function::Dirac(
        f0, tshift, factor, use_trick_for_better_pressure);
  }
};

specfem::sources::moment_tensor::moment_tensor(
    specfem::utilities::moment_tensor &moment_tensor)
    : x(moment_tensor.x), z(moment_tensor.z), Mxx(moment_tensor.Mxx),
      Mxz(moment_tensor.Mxz), Mzz(moment_tensor.Mzz) {

  if (moment_tensor.stf_type == "Dirac") {
    this->forcing_function = new specfem::forcing_function::Dirac(
        moment_tensor.f0, moment_tensor.tshift, moment_tensor.factor, true);
  }
};

void specfem::sources::force::compute_stf(
    specfem::HostView1d<type_real, LayoutStride> stf_array,
    specfem::TimeScheme::TimeScheme *it) {

  while (it->status()) {
    type_real timeval = it->get_time();
    int istep = it->get_timestep();
    stf_array(istep) = this->forcing_function->compute(timeval);

    it->update_time();
  }

  it->reset_time();
};

void specfem::sources::moment_tensor::compute_stf(
    specfem::HostView1d<type_real, LayoutStride> stf_array,
    specfem::TimeScheme::TimeScheme *it) {

  while (it->status()) {
    type_real timeval = it->get_time();
    int istep = it->get_timestep();
    stf_array(istep) = this->forcing_function->compute(timeval);

    it->update_time();
  }

  it->reset_time();
};

void specfem::sources::source::print(std::ostream &out) const {
  out << "Error allocating source. Base class being called.";

  throw std::runtime_error("Error allocating source. Base class being called.");

  return;
}

void specfem::sources::force::print(std::ostream &out) const {
  out << "Source Information: Force Source \n"
      << "   Source Location: \n"
      << "                    x = " << this->x << "\n"
      << "                    z = " << this->z << "\n"
      << "                    xi = " << this->xi << "\n"
      << "                    gamma = " << this->gamma << "\n"
      << "                    ispec = " << this->ispec << "\n"
      << "                    islice = " << this->islice << "\n";
  out << *(this->forcing_function);

  return;
}

void specfem::sources::moment_tensor::print(std::ostream &out) const {
  out << "Source Information: Moment Tensor Source \n"
      << "   Source Location: \n"
      << "                    x = " << this->x << "\n"
      << "                    z = " << this->z << "\n"
      << "                    xi = " << this->xi << "\n"
      << "                    gamma = " << this->gamma << "\n"
      << "                    ispec = " << this->ispec << "\n"
      << "                    islice = " << this->islice << "\n";
  out << *(this->forcing_function);

  return;
}

std::ostream &
specfem::sources::operator<<(std::ostream &out,
                             const specfem::sources::source &source) {
  source.print(out);
  return out;
}
