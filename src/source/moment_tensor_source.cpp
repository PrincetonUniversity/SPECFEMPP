#include "algorithms/interface.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/properties/properties.hpp"
#include "globals.h"
#include "jacobian/interface.hpp"
#include "kokkos_abstractions.h"
#include "point/coordinates.hpp"
#include "point/partial_derivatives.hpp"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
// #include "utilities.cpp"
#include "yaml-cpp/yaml.h"
#include <cmath>

void specfem::sources::moment_tensor::compute_source_array(
    const specfem::compute::mesh &mesh,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      this->x, this->z);
  auto lcoord = specfem::algorithms::locate_point(coord, mesh);

  const auto el_type = element_types.get_medium_tag(lcoord.ispec);

  if (el_type == specfem::element::medium_tag::acoustic) {
    throw std::runtime_error(
        "Moment tensor source not implemented for acoustic medium");
  }

  const int ncomponents = source_array.extent(0);
  if (ncomponents != 2) {
    throw std::runtime_error(
        "Moment tensor source requires 2 components for elastic medium");
  }

  const auto xi = mesh.quadratures.gll.h_xi;
  const auto gamma = mesh.quadratures.gll.h_xi;
  const auto N = mesh.quadratures.gll.N;

  // Compute lagrange interpolants at the source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.xi, N, xi);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.gamma, N, gamma);

  // Load
  specfem::kokkos::HostView2d<type_real> source_polynomial("source_polynomial",
                                                           N, N);
  using PointPartialDerivatives =
      specfem::point::partial_derivatives<specfem::dimension::type::dim2, false,
                                          false>;
  specfem::kokkos::HostView2d<PointPartialDerivatives> element_derivatives(
      "element_derivatives", N, N);

  Kokkos::parallel_for(
      specfem::kokkos::HostMDrange<2>({ 0, 0 }, { N, N }),
      // Structured binding does not work with lambdas
      // Workaround: capture by value
      [=, hxi_source = hxi_source,
       hgamma_source = hgamma_source](const int iz, const int ix) {
        type_real hlagrange = hxi_source(ix) * hgamma_source(iz);
        const specfem::point::index<specfem::dimension::type::dim2> index(
            lcoord.ispec, iz, ix);
        PointPartialDerivatives derivatives;
        specfem::compute::load_on_host(index, partial_derivatives, derivatives);
        source_polynomial(iz, ix) = hlagrange;
        element_derivatives(iz, ix) = derivatives;
      });

  // Store the derivatives in a function object for interpolation
  auto derivatives_source = specfem::algorithms::interpolate_function(
      source_polynomial, element_derivatives);

  // type_real hlagrange;
  // type_real dxis_dx = 0;
  // type_real dxis_dz = 0;
  // type_real dgammas_dx = 0;
  // type_real dgammas_dz = 0;

  // for (int iz = 0; iz < N; iz++) {
  //   for (int ix = 0; ix < N; ix++) {
  //     auto derivatives =
  //         partial_derivatives
  //             .load_derivatives<false, specfem::kokkos::HostExecSpace>(
  //                 lcoord.ispec, j, i);
  //     hlagrange = hxis(i) * hgammas(j);
  //     dxis_dx += hlagrange * derivatives.xix;
  //     dxis_dz += hlagrange * derivatives.xiz;
  //     dgammas_dx += hlagrange * derivatives.gammax;
  //     dgammas_dz += hlagrange * derivatives.gammaz;
  //   }
  // }

  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      type_real dsrc_dx =
          (hpxi_source(ix) * derivatives_source.xix) * hgamma_source(iz) +
          hxi_source(ix) * (hpgamma_source(iz) * derivatives_source.gammax);
      type_real dsrc_dz =
          (hpxi_source(ix) * derivatives_source.xiz) * hgamma_source(iz) +
          hxi_source(ix) * (hpgamma_source(iz) * derivatives_source.gammaz);
      if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
        source_array(0, iz, ix) += Mxx * dsrc_dx + Mxz * dsrc_dz;
        source_array(1, iz, ix) += Mxz * dsrc_dx + Mzz * dsrc_dz;
      } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
        source_array(0, iz, ix) += Mxx * dsrc_dx + Mxz * dsrc_dz;
        source_array(1, iz, ix) += 0;
      }
    }
  }
}

std::string specfem::sources::moment_tensor::print() const {
  std::ostringstream message;
  message << "- Moment Tensor Source: \n"
          << "    Source Location: \n"
          << "      x = " << this->x << "\n"
          << "      z = " << this->z << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";
  // out << *(this->forcing_function);

  return message.str();
}
