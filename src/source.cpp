#include "../include/source.h"
#include "../include/config.h"
#include "../include/globals.h"
#include "../include/jacobian.h"
#include "../include/kokkos_abstractions.h"
#include "../include/lagrange_poly.h"
#include "../include/source_time_function.h"
#include "../include/specfem_mpi.h"
#include "../include/utils.h"
#include "yaml-cpp/yaml.h"
#include <cmath>

KOKKOS_IMPL_HOST_FUNCTION
specfem::forcing_function::stf *assign_stf(std::string forcing_type,
                                           type_real f0, type_real tshift,
                                           type_real factor, type_real dt,
                                           bool use_trick_for_better_pressure) {

  specfem::forcing_function::stf *forcing_function;
  if (forcing_type == "Dirac") {
    forcing_function = (specfem::forcing_function::stf *)
        Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(
            sizeof(specfem::forcing_function::Dirac));

    f0 = 1.0 / (10.0 * dt);

    Kokkos::parallel_for(
        "specfem::sources::moment_tensor::moment_tensor::allocate_stf",
        specfem::kokkos::DeviceRange(0, 1), KOKKOS_LAMBDA(const int &) {
          new (forcing_function) specfem::forcing_function::Dirac(
              f0, tshift, factor, use_trick_for_better_pressure);
        });

    Kokkos::fence();
  }

  return forcing_function;
}

KOKKOS_IMPL_HOST_FUNCTION
specfem::forcing_function::stf *
assign_dirac(YAML::Node &Dirac, type_real dt,
             bool use_trick_for_better_pressure) {

  specfem::forcing_function::stf *forcing_function;
  forcing_function = (specfem::forcing_function::stf *)
      Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(
          sizeof(specfem::forcing_function::Dirac));

  type_real f0 = 1.0 / (10.0 * dt);
  type_real tshift = Dirac["tshift"].as<type_real>();
  type_real factor = Dirac["factor"].as<type_real>();

  Kokkos::parallel_for(
      "specfem::sources::moment_tensor::moment_tensor::allocate_stf",
      specfem::kokkos::DeviceRange(0, 1), KOKKOS_LAMBDA(const int &) {
        new (forcing_function) specfem::forcing_function::Dirac(
            f0, tshift, factor, use_trick_for_better_pressure);
      });

  Kokkos::fence();

  return forcing_function;
}

void specfem::sources::source::check_locations(const type_real xmin,
                                               const type_real xmax,
                                               const type_real zmin,
                                               const type_real zmax,
                                               const specfem::MPI::MPI *mpi) {
  specfem::utilities::check_locations(this->get_x(), this->get_z(), xmin, xmax,
                                      zmin, zmax, mpi);
}

void specfem::sources::force::locate(
    const specfem::kokkos::HostView2d<type_real> coord,
    const specfem::kokkos::HostMirror3d<int> h_ibool,
    const specfem::kokkos::HostMirror1d<type_real> xigll,
    const specfem::kokkos::HostMirror1d<type_real> zigll, const int nproc,
    const specfem::kokkos::HostView2d<type_real> coorg,
    const specfem::kokkos::HostView2d<int> knods, const int npgeo,
    const specfem::kokkos::HostMirror1d<specfem::elements::type> ispec_type,
    const specfem::MPI::MPI *mpi) {
  std::tie(this->xi, this->gamma, this->ispec, this->islice) =
      specfem::utilities::locate(coord, h_ibool, xigll, zigll, nproc,
                                 this->get_x(), this->get_z(), coorg, knods,
                                 npgeo, mpi);
  if (this->islice == mpi->get_rank()) {
    this->el_type = ispec_type(ispec);
  }
}

void specfem::sources::moment_tensor::locate(
    const specfem::kokkos::HostView2d<type_real> coord,
    const specfem::kokkos::HostMirror3d<int> h_ibool,
    const specfem::kokkos::HostMirror1d<type_real> xigll,
    const specfem::kokkos::HostMirror1d<type_real> zigll, const int nproc,
    const specfem::kokkos::HostView2d<type_real> coorg,
    const specfem::kokkos::HostView2d<int> knods, const int npgeo,
    const specfem::kokkos::HostMirror1d<specfem::elements::type> ispec_type,
    const specfem::MPI::MPI *mpi) {
  std::tie(this->xi, this->gamma, this->ispec, this->islice) =
      specfem::utilities::locate(coord, h_ibool, xigll, zigll, nproc,
                                 this->get_x(), this->get_z(), coorg, knods,
                                 npgeo, mpi);

  if (this->islice == mpi->get_rank()) {
    if (ispec_type(ispec) != specfem::elements::elastic) {
      throw std::runtime_error(
          "Found a Moment-tensor source in acoustic/poroelastic element");
    } else {
      this->el_type = specfem::elements::elastic;
    }
  }
  int ngnod = knods.extent(0);
  this->s_coorg = specfem::kokkos::HostView2d<type_real>(
      "specfem::sources::moment_tensor::s_coorg", ndim, ngnod);

  // Store s_coorg for better caching
  for (int in = 0; in < ngnod; in++) {
    this->s_coorg(0, in) = coorg(0, knods(in, ispec));
    this->s_coorg(1, in) = coorg(1, knods(in, ispec));
  }

  return;
}

void specfem::sources::force::compute_source_array(
    const specfem::quadrature::quadrature &quadx,
    const specfem::quadrature::quadrature &quadz,
    specfem::kokkos::HostView3d<type_real> source_array) {

  type_real xi = this->xi;
  type_real gamma = this->gamma;
  type_real angle = this->angle;
  specfem::elements::type el_type = this->el_type;

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

      if (el_type == specfem::elements::acoustic ||
          (el_type == specfem::elements::elastic &&
           specfem::globals::simulation_wave == specfem::wave::sh)) {
        source_array(j, i, 0) = hlagrange;
        source_array(j, i, 1) = hlagrange;
      } else if ((el_type == specfem::elements::elastic &&
                  specfem::globals::simulation_wave == specfem::wave::p_sv) ||
                 el_type == specfem::elements::poroelastic) {
        type_real tempx = sin(angle) * hlagrange;
        source_array(j, i, 0) = tempx;
        type_real tempz = -1.0 * cos(angle) * hlagrange;
        source_array(j, i, 1) = tempz;
      }
    }
  }
};

void specfem::sources::moment_tensor::compute_source_array(
    const specfem::quadrature::quadrature &quadx,
    const specfem::quadrature::quadrature &quadz,
    specfem::kokkos::HostView3d<type_real> source_array) {

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
  // ToDo:: Need to implement a check to see if acoustic source lies on an
  // acoustic surface
}

void specfem::sources::moment_tensor::check_locations(
    const type_real xmin, const type_real xmax, const type_real zmin,
    const type_real zmax, const specfem::MPI::MPI *mpi) {

  specfem::utilities::check_locations(this->get_x(), this->get_z(), xmin, xmax,
                                      zmin, zmax, mpi);
  // ToDo:: Need to implement a check to see if acoustic source lies on an
  // acoustic surface
}

// specfem::sources::force::force(type_real x, type_real z, type_real angle,
//                                type_real tshift, type_real f0, type_real
//                                factor, const type_real dt, std::string
//                                forcing_type, wave_type wave)
//     : x(x), z(z), angle(angle), wave(wave) {

//   bool use_trick_for_better_pressure = false;

//   this->forcing_function = assign_stf(forcing_type, f0, tshift, factor, dt,
//                                       use_trick_for_better_pressure);
// };

specfem::sources::force::force(YAML::Node &Node, const type_real dt)
    : x(Node["x"].as<type_real>()), z(Node["z"].as<type_real>()),
      angle(Node["angle"].as<type_real>()) {

  bool use_trick_for_better_pressure = false;

  if (YAML::Node Dirac = Node["Dirac"]) {
    this->forcing_function =
        assign_dirac(Dirac, dt, use_trick_for_better_pressure);
  }
};

// specfem::sources::moment_tensor::moment_tensor(type_real x, type_real z,
//                                                type_real Mxx, type_real Mxz,
//                                                type_real Mzz, type_real
//                                                tshift, type_real f0,
//                                                type_real factor, const
//                                                type_real dt, std::string
//                                                forcing_type)
//     : x(x), z(z), Mxx(Mxx), Mxz(Mxz), Mzz(Mzz) {

//   bool use_trick_for_better_pressure = false;

//   this->forcing_function = assign_stf(forcing_type, f0, tshift, factor, dt,
//                                       use_trick_for_better_pressure);
// };

specfem::sources::moment_tensor::moment_tensor(YAML::Node &Node,
                                               const type_real dt)
    : x(Node["x"].as<type_real>()), z(Node["z"].as<type_real>()),
      Mxx(Node["Mxx"].as<type_real>()), Mxz(Node["Mxz"].as<type_real>()),
      Mzz(Node["Mzz"].as<type_real>()) {

  bool use_trick_for_better_pressure = false;

  if (YAML::Node Dirac = Node["Dirac"]) {
    this->forcing_function =
        assign_dirac(Dirac, dt, use_trick_for_better_pressure);
  }
};

KOKKOS_IMPL_HOST_FUNCTION
type_real specfem::sources::force::get_t0() const {

  type_real t0;
  specfem::forcing_function::stf *forcing_func = this->forcing_function;

  Kokkos::parallel_reduce(
      "specfem::sources::force::get_t0", specfem::kokkos::DeviceRange({ 0, 1 }),
      KOKKOS_LAMBDA(const int &, type_real &lsum) {
        lsum = forcing_func->get_t0();
      },
      t0);

  Kokkos::fence();

  return t0;
}

KOKKOS_IMPL_HOST_FUNCTION
type_real specfem::sources::moment_tensor::get_t0() const {

  type_real t0;
  specfem::forcing_function::stf *forcing_func = this->forcing_function;

  Kokkos::parallel_reduce(
      "specfem::sources::force::get_t0", specfem::kokkos::DeviceRange({ 0, 1 }),
      KOKKOS_LAMBDA(const int &, type_real &lsum) {
        lsum = forcing_func->get_t0();
      },
      t0);

  Kokkos::fence();

  return t0;
}

void specfem::sources::force::update_tshift(type_real tshift) {

  specfem::forcing_function::stf *forcing_func = this->forcing_function;

  Kokkos::parallel_for(
      "specfem::sources::force::get_t0", specfem::kokkos::DeviceRange({ 0, 1 }),
      KOKKOS_LAMBDA(const int &) { forcing_func->update_tshift(tshift); });

  Kokkos::fence();

  return;
}

void specfem::sources::moment_tensor::update_tshift(type_real tshift) {

  specfem::forcing_function::stf *forcing_func = this->forcing_function;

  Kokkos::parallel_for(
      "specfem::sources::moment_tensor::get_t0",
      specfem::kokkos::DeviceRange({ 0, 1 }),
      KOKKOS_LAMBDA(const int &) { forcing_func->update_tshift(tshift); });

  Kokkos::fence();

  return;
}

void specfem::sources::source::print(std::ostream &out) const {
  out << "Error allocating source. Base class being called.";

  throw std::runtime_error("Error allocating source. Base class being called.");

  return;
}

void specfem::sources::force::print(std::ostream &out) const {
  out << "Force Source: \n"
      << "   Source Location: \n"
      << "    x = " << this->x << "\n"
      << "    z = " << this->z << "\n"
      << "    xi = " << this->xi << "\n"
      << "    gamma = " << this->gamma << "\n"
      << "    ispec = " << this->ispec << "\n"
      << "    islice = " << this->islice << "\n";
  // out << *(this->forcing_function);

  return;
}

std::string specfem::sources::force::print() const {
  std::ostringstream message;
  message << "- Force Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "      xi = " << this->xi << "\n"
          << "      gamma = " << this->gamma << "\n"
          << "      ispec = " << this->ispec << "\n"
          << "      islice = " << this->islice << "\n";

  return message.str();
}

void specfem::sources::moment_tensor::print(std::ostream &out) const {
  out << "Moment Tensor Source: \n"
      << "   Source Location: \n"
      << "    x = " << this->x << "\n"
      << "    z = " << this->z << "\n"
      << "    xi = " << this->xi << "\n"
      << "    gamma = " << this->gamma << "\n"
      << "    ispec = " << this->ispec << "\n"
      << "    islice = " << this->islice << "\n";
  // out << *(this->forcing_function);

  return;
}

std::string specfem::sources::moment_tensor::print() const {
  std::ostringstream message;
  message << "- Moment Tensor Source: \n"
          << "    Source Location: \n"
          << "      x = " << this->x << "\n"
          << "      z = " << this->z << "\n"
          << "      xi = " << this->xi << "\n"
          << "      gamma = " << this->gamma << "\n"
          << "      ispec = " << this->ispec << "\n"
          << "      islice = " << this->islice << "\n";
  // out << *(this->forcing_function);

  return message.str();
  ;
}

std::ostream &
specfem::sources::operator<<(std::ostream &out,
                             const specfem::sources::source &source) {
  source.print(out);
  return out;
}
