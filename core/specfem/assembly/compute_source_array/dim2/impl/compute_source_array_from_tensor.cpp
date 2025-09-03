#include "compute_source_array_from_tensor.hpp"
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
#include <Kokkos_Core.hpp>

// Local namespace for implementation details
namespace specfem::assembly::compute_source_array_impl {

void compute_source_array_from_tensor_and_element_jacobian(
    const specfem::sources::tensor_source<specfem::dimension::type::dim2>
        &tensor_source,
    const JacobianViewType &element_jacobian_matrix,
    const specfem::assembly::mesh_impl::quadrature<
        specfem::dimension::type::dim2> &quadrature,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array) {

  const int ngllx = quadrature.N;
  const int ngllz = quadrature.N;

  // Create quadrature and compute xi/gamma arrays
  auto xi = quadrature.h_xi;
  auto gamma = quadrature.h_xi;

  // Compute lagrange interpolants at the local source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          tensor_source.get_local_coordinates().xi, ngllx, xi);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          tensor_source.get_local_coordinates().gamma, ngllz, gamma);

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

void specfem::assembly::compute_source_array_impl::from_tensor(
    const specfem::sources::tensor_source<specfem::dimension::type::dim2>
        &tensor_source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array) {

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
          tensor_source.get_local_coordinates().ispec, iz, ix);
      PointJacobianMatrix derivatives;
      specfem::assembly::load_on_host(index, jacobian_matrix, derivatives);
      element_jacobian(iz, ix) = derivatives;
    }
  }
  const auto &quadrature =
      static_cast<const specfem::assembly::mesh_impl::quadrature<
          specfem::dimension::type::dim2> &>(mesh);

  specfem::assembly::compute_source_array_impl::
      compute_source_array_from_tensor_and_element_jacobian(
          tensor_source, element_jacobian, quadrature, source_array);

  return;
}
