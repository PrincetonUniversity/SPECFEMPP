#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities.cpp"
#include "utils.h"
#include "yaml-cpp/yaml.h"
#include <cmath>

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

void specfem::sources::force::compute_source_array(
    const specfem::quadrature::quadrature *quadx,
    const specfem::quadrature::quadrature *quadz,
    specfem::kokkos::HostView3d<type_real> source_array) {

  type_real xi = this->xi;
  type_real gamma = this->gamma;
  type_real angle = this->angle;
  specfem::elements::type el_type = this->el_type;

  auto [hxis, hpxis] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          xi, quadx->get_N(), quadx->get_hxi());
  auto [hgammas, hpgammas] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          gamma, quadz->get_N(), quadz->get_hxi());

  int nquadx = quadx->get_N();
  int nquadz = quadz->get_N();

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

specfem::sources::force::force(YAML::Node &Node, const type_real dt)
    : x(Node["x"].as<type_real>()), z(Node["z"].as<type_real>()),
      angle(Node["angle"].as<type_real>()) {

  bool use_trick_for_better_pressure = false;

  if (YAML::Node Dirac = Node["Dirac"]) {
    this->forcing_function =
        assign_dirac(Dirac, dt, use_trick_for_better_pressure);
  }
};

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

void specfem::sources::force::update_tshift(type_real tshift) {

  specfem::forcing_function::stf *forcing_func = this->forcing_function;

  Kokkos::parallel_for(
      "specfem::sources::force::get_t0", specfem::kokkos::DeviceRange({ 0, 1 }),
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
