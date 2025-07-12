#include "adjoint_source_array.hpp"
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

/**
 * @brief Specialization of compute_source_array for adjoint sources in 2D
 *
 * This function computes the source array for an adjoint source in 2D.
 * It uses the GLL quadrature points to compute the source array based on the
 * specified parameters of the adjoint source.
 */
bool specfem::assembly::compute_source_array_impl::adjoint_source_array(
    const std::shared_ptr<specfem::sources::source> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  // Check if the source is correct type
  if (source->get_source_type() !=
      specfem::sources::source_type::adjoint_source)
    return false;

  // Cast to derived class to access specific methods
  auto adjoint_source =
      static_cast<const specfem::sources::adjoint_source *>(source.get());

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      adjoint_source->get_x(), adjoint_source->get_z());
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

  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      hlagrange = hxi_source(ix) * hgamma_source(iz);

      if (el_type == specfem::element::medium_tag::acoustic) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "Adjoint source requires 1 component for acoustic medium");
        }
        source_array(0, iz, ix) = hlagrange;
      }
      // Elastic SH
      else if (el_type == specfem::element::medium_tag::elastic_sh) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "Adjoint source requires 1 component for elastic SH medium");
        }
        source_array(0, iz, ix) = hlagrange;
      } else if (el_type == specfem::element::medium_tag::elastic_psv) {
        if (ncomponents != 2) {
          throw std::runtime_error(
              "Adjoint source for elastic P-SV, poroelastic, or "
              "electromagnetic P-SV requires 2 components");
        }
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = hlagrange;
      } else if ((el_type == specfem::element::medium_tag::poroelastic)) {
        if (ncomponents != 4) {
          throw std::runtime_error(
              "Force source requires 4 components for poroelastic medium");
        }
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = hlagrange;
        source_array(2, iz, ix) = hlagrange;
        source_array(3, iz, ix) = hlagrange;
      } else if (el_type == specfem::element::medium_tag::elastic_psv_t) {
        if (ncomponents != 3) {
          throw std::runtime_error(
              "Adjoint source requires 3 components for elastic psv_t medium");
        }
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = hlagrange;
        source_array(2, iz, ix) = static_cast<type_real>(0.0);
      } else if (el_type == specfem::element::medium_tag::electromagnetic_te) {
        if (ncomponents != 2) {
          throw std::runtime_error(
              "Adjoint source requires 2 components for electromagnetic "
              "TE medium");
        }
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = hlagrange;
      } else {
        KOKKOS_ABORT_WITH_LOCATION("Adjoint source array computation not "
                                   "implemented for requested element type.");
      }
    }
  }
  return true;
}
