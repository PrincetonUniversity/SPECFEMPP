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

  if (el_type == specfem::element::medium_tag::elastic_sh) {
    throw std::runtime_error(
        "Moment tensor source not implemented for elastic SH medium");
  }

  const int ncomponents = source_array.extent(0);
  if ((el_type == specfem::element::medium_tag::elastic_psv) ||
      (el_type == specfem::element::medium_tag::electromagnetic_te)) {
    if (ncomponents != 2) {
      throw std::runtime_error(
          "Moment tensor source requires 2 components for elastic medium");
    }
  } else if (el_type == specfem::element::medium_tag::poroelastic) {
    if (ncomponents != 4) {
      throw std::runtime_error(
          "Moment tensor source requires 4 components for poroelastic medium");
    }
  } else if (el_type == specfem::element::medium_tag::elastic_psv_t) {
    if (ncomponents != 3) {
      throw std::runtime_error("Moment tensor source requires 3 components for "
                               "elastic psv_t medium");
    }
  } else {
    std::ostringstream message;
    message << "Moment tensor source not implemented for element type: "
            << specfem::element::to_string(el_type);
    auto message_str = message.str();
    Kokkos::abort(message_str.c_str());
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
      source_array(0, iz, ix) = Mxx * dsrc_dx + Mxz * dsrc_dz;
      source_array(1, iz, ix) = Mxz * dsrc_dx + Mzz * dsrc_dz;

      if (el_type == specfem::element::medium_tag::poroelastic) {
        source_array(2, iz, ix) = source_array(0, iz, ix);
        source_array(3, iz, ix) = source_array(1, iz, ix);
      } else if (el_type == specfem::element::medium_tag::elastic_psv_t) {
        source_array(2, iz, ix) = static_cast<type_real>(0.0);
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
          << "    Moment Tensor: \n"
          << "      Mxx, Mzz, Mxz = " << this->Mxx << ", " << this->Mzz << ", "
          << this->Mxz << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}

bool specfem::sources::moment_tensor::operator==(
    const specfem::sources::source &other) const {

  // Try casting the other source to a moment tensor source
  const auto *other_source =
      dynamic_cast<const specfem::sources::moment_tensor *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a moment tensor object" << std::endl;
    return false;
  }

  bool internal =
      specfem::utilities::almost_equal(this->Mxx, other_source->Mxx) &&
      specfem::utilities::almost_equal(this->Mxz, other_source->Mxz) &&
      specfem::utilities::almost_equal(this->Mzz, other_source->Mzz) &&
      specfem::utilities::almost_equal(this->x, other_source->x) &&
      specfem::utilities::almost_equal(this->z, other_source->z);

  if (!internal) {
    std::cout << "Moment tensor source not equal" << std::endl;
  }

  return internal &&
         (*(this->forcing_function) == *(other_source->forcing_function));
}
bool specfem::sources::moment_tensor::operator!=(
    const specfem::sources::source &other) const {
  return !(*this == other);
}
