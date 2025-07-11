

#include "cosserat_force_source_array.hpp"
#include "algorithms/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
/**
 * @brief Compute the source array for a Cosserat force source
 *
 * This function computes the source array for a Cosserat force source in 2D.
 * It uses the GLL quadrature points to compute the source array based on the
 * specified parameters of the Cosserat force source.
 */
bool specfem::assembly::compute_source_array_impl::cosserat_force_source_array(
    const std::shared_ptr<specfem::sources::source> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  // Check if the source is correct type
  if (source->get_source_type() !=
      specfem::sources::source_type::cosserat_force_source) {
    return false;
  }

  // Cast to derived class to access specific methods
  auto cosserat_source =
      static_cast<const specfem::sources::cosserat_force *>(source.get());

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      cosserat_source->get_x(), cosserat_source->get_z());

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

  if (el_type != specfem::element::medium_tag::elastic_psv_t) {
    std::ostringstream message;
    message << "Source array computation not implemented for element type: "
            << specfem::element::to_string(el_type) << "\n\texpected: "
            << specfem::element::to_string(
                   specfem::element::medium_tag::elastic_psv_t)
            << " [" << __FILE__ << ":" << __LINE__ << "]";
    auto message_str = message.str();
    throw std::runtime_error(message_str);
  }

  if (ncomponents != 3) {
    throw std::runtime_error(
        "Source array requires 3 components for elastic psv_t medium");
  }

  // Source array computation
  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      hlagrange = hxi_source(ix) * hgamma_source(iz);

      source_array(0, iz, ix) = cosserat_source->get_f() *
                                std::sin(Kokkos::numbers::pi_v<type_real> /
                                         180 * cosserat_source->get_angle()) *
                                hlagrange;
      source_array(1, iz, ix) = -1.0 * cosserat_source->get_f() *
                                std::cos(Kokkos::numbers::pi_v<type_real> /
                                         180 * cosserat_source->get_angle()) *
                                hlagrange;

      source_array(2, iz, ix) = cosserat_source->get_fc() * hlagrange;
    }
  }

  return true;
}
