#include "moment_tensor_source_array.hpp"
#include "algorithms/interface.hpp"
#include "enumerations/macros.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

bool specfem::assembly::compute_source_array_impl::moment_tensor_source_array(
    const std::shared_ptr<specfem::sources::source> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  // Check if the source is correct type
  if (source->get_source_type() !=
      specfem::sources::source_type::moment_tensor_source) {
    return false;
  }

  // Cast to derived class to access specific methods
  auto moment_tensor_source =
      static_cast<const specfem::sources::moment_tensor *>(source.get());

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      moment_tensor_source->get_x(), moment_tensor_source->get_z());
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
    KOKKOS_ABORT_WITH_LOCATION("Moment tensor source array computation not "
                               "implemented for requested element type.");
  }

  const auto xi = mesh.h_xi;
  const auto gamma = mesh.h_xi;
  const auto N = mesh.ngllx;

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
  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::dimension::type::dim2, false,
                                      false>;
  specfem::kokkos::HostView2d<PointJacobianMatrix> element_derivatives(
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
        PointJacobianMatrix derivatives;
        specfem::assembly::load_on_host(index, jacobian_matrix, derivatives);
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
  //         jacobian_matrix
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
      source_array(0, iz, ix) = moment_tensor_source->get_Mxx() * dsrc_dx +
                                moment_tensor_source->get_Mxz() * dsrc_dz;
      source_array(1, iz, ix) = moment_tensor_source->get_Mxz() * dsrc_dx +
                                moment_tensor_source->get_Mzz() * dsrc_dz;

      if (el_type == specfem::element::medium_tag::poroelastic) {
        source_array(2, iz, ix) = source_array(0, iz, ix);
        source_array(3, iz, ix) = source_array(1, iz, ix);
      } else if (el_type == specfem::element::medium_tag::elastic_psv_t) {
        source_array(2, iz, ix) = static_cast<type_real>(0.0);
      }
    }
  }

  return true;
}
