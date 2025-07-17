#include "algorithms/interface.hpp"
#include "enumerations/macros.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"

template <>
void specfem::assembly::compute_source_array<specfem::dimension::type::dim2>(
    const specfem::sources::tensor_source &tensor_source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    specfem::kokkos::HostView3d<type_real> source_array) {

  // Not getting around the mesh input
  auto xi = mesh.h_xi;
  auto gamma = mesh.h_xi;
  const int ngllx = mesh.ngllx;
  const int ngllz = mesh.ngllz;

  // Compute lagrange interpolants at the local source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          tensor_source.get_xi(), ngllx, xi);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          tensor_source.get_gamma(), ngllz, gamma);

  specfem::kokkos::HostView2d<type_real> source_polynomial("source_polynomial",
                                                           ngllz, ngllx);
  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::dimension::type::dim2, false,
                                      false>;
  specfem::kokkos::HostView2d<PointJacobianMatrix> element_derivatives(
      "element_derivatives", ngllz, ngllx);

  Kokkos::parallel_for(
      specfem::kokkos::HostMDrange<2>({ 0, 0 }, { ngllz, ngllx }),
      // Structured binding does not work with lambdas
      // Workaround: capture by value
      [=, hxi_source = hxi_source, hgamma_source = hgamma_source,
       &tensor_source](const int iz, const int ix) {
        type_real hlagrange = hxi_source(ix) * hgamma_source(iz);
        const specfem::point::index<specfem::dimension::type::dim2> index(
            tensor_source.get_element_index(), iz, ix);
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

  const auto source_tensor = tensor_source.get_source_tensor();

  int ncomponents = source_array.extent(0);

  // Sanity check
  if (ncomponents != source_tensor.extent(0)) {
    KOKKOS_ABORT_WITH_LOCATION(
        "source_array_components and tensor components do not match")
  }

  for (int iz = 0; iz < ngllz; ++iz) {
    for (int ix = 0; ix < ngllx; ++ix) {

      // Compute the derivatives at the source location
      type_real dsrc_dx =
          (hpxi_source(ix) * derivatives_source.xix) * hgamma_source(iz) +
          hxi_source(ix) * (hpgamma_source(iz) * derivatives_source.gammax);
      type_real dsrc_dz =
          (hpxi_source(ix) * derivatives_source.xiz) * hgamma_source(iz) +
          hxi_source(ix) * (hpgamma_source(iz) * derivatives_source.gammaz);

      for (int i = 0; i < ncomponents; ++i) {
        source_array(i, iz, ix) =
            source_tensor(i, 0) * dsrc_dx + source_tensor(i, 1) * dsrc_dz;
      }
    }
  }

  return;
}
