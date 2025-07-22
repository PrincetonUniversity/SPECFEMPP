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

// Local namespace for implementation details
namespace specfem::assembly::compute_source_array_impl {

using PointJacobianMatrix =
    specfem::point::jacobian_matrix<specfem::dimension::type::dim2, false,
                                    false>;
using JacobianViewType = specfem::kokkos::HostView2d<PointJacobianMatrix>;

void compute_source_array_from_tensor_and_element_jacobian(
    const specfem::sources::tensor_source<specfem::dimension::type::dim2>
        &tensor_source,
    const JacobianViewType &element_jacobian_matrix,
    specfem::kokkos::HostView3d<type_real> source_array) {

  const int ngllx = source_array.extent(2);
  const int ngllz = source_array.extent(1);

  // Create quadrature and compute xi/gamma arrays
  specfem::quadrature::gll::gll quadrature_x(0.0, 0.0, ngllx);
  specfem::quadrature::gll::gll quadrature_z(0.0, 0.0, ngllz);
  auto xi = quadrature_x.get_hxi();
  auto gamma = quadrature_z.get_hxi();

  // Compute lagrange interpolants at the local source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          tensor_source.get_xi(), ngllx, xi);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          tensor_source.get_gamma(), ngllz, gamma);

  specfem::kokkos::HostView2d<type_real> source_polynomial("source_polynomial",
                                                           ngllz, ngllx);

  // Use pre-computed jacobian data instead of loading from jacobian_matrix
  for (int iz = 0; iz < ngllz; ++iz) {
    for (int ix = 0; ix < ngllx; ++ix) {
      type_real hlagrange = hxi_source(ix) * hgamma_source(iz);
      source_polynomial(iz, ix) = hlagrange;
    }
  }

  // Store the derivatives in a function object for interpolation
  auto derivatives_source = specfem::algorithms::interpolate_function(
      source_polynomial, element_jacobian_matrix);

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

} // namespace specfem::assembly::compute_source_array_impl

template <>
void specfem::assembly::compute_source_array_impl::from_tensor<
    specfem::dimension::type::dim2>(
    const specfem::sources::tensor_source<specfem::dimension::type::dim2>
        &tensor_source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    specfem::kokkos::HostView3d<type_real> source_array) {

  const int ngllx = source_array.extent(2);
  const int ngllz = source_array.extent(1);

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::dimension::type::dim2, false,
                                      false>;
  specfem::kokkos::HostView2d<PointJacobianMatrix> element_jacobian(
      "element_jacobian", ngllz, ngllx);

  // Extract jacobian data from jacobian_matrix
  for (int iz = 0; iz < ngllz; ++iz) {
    for (int ix = 0; ix < ngllx; ++ix) {
      const specfem::point::index<specfem::dimension::type::dim2> index(
          tensor_source.get_element_index(), iz, ix);
      PointJacobianMatrix derivatives;
      specfem::assembly::load_on_host(index, jacobian_matrix, derivatives);
      element_jacobian(iz, ix) = derivatives;
    }
  }

  specfem::assembly::compute_source_array_impl::
      compute_source_array_from_tensor_and_element_jacobian(
          tensor_source, element_jacobian, source_array);

  return;
}
