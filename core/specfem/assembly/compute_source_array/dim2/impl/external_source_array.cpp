#include "external_source_array.hpp"
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

bool specfem::assembly::compute_source_array_impl::external_source_array(
    const std::shared_ptr<specfem::sources::source> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  // Check if the source is correct type
  if (source->get_source_type() !=
      specfem::sources::source_type::external_source) {
    return false;
  }

  // Cast to derived class to access specific methods
  auto external_source =
      static_cast<const specfem::sources::external *>(source.get());

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      external_source->get_x(), external_source->get_z());
  auto lcoord = specfem::algorithms::locate_point(coord, mesh);

  const auto xi = mesh.h_xi;
  const auto gamma = mesh.h_xi;
  const auto N = mesh.ngllx;

  const auto el_type = element_types.get_medium_tag(lcoord.ispec);
  const int ncomponents = source_array.extent(0);

  // Compute lagrange interpolants at the source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.xi, N, xi);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.gamma, N, gamma);

  type_real hlagrange;

  const auto force_vector = external_source->get_force_vector();

  // Source array computation
  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      hlagrange = hxi_source(ix) * hgamma_source(iz);
      for (int i = 0; i < ncomponents; ++i) {
        source_array(i, iz, ix) = hlagrange * force_vector(i);
      }
    }
  }

  return true;
}
