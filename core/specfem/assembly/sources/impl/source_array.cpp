#include "source_array.hpp"
#include "algorithms/interface.hpp"
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
void specfem::assembly::sources_impl::compute_source_array(
    const std::shared_ptr<specfem::sources::adjoint_source> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      source->get_x(), source->get_z());
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
        std::ostringstream message;
        message << "Source array computation not implemented for element type: "
                << specfem::element::to_string(el_type) << " [" << __FILE__
                << ":" << __LINE__ << "]";
        auto message_str = message.str();
        Kokkos::abort(message_str.c_str());
      }
    }
  }
}

/**
 * @brief Compute the source array for a Cosserat force source
 *
 * This function computes the source array for a Cosserat force source in 2D.
 * It uses the GLL quadrature points to compute the source array based on the
 * specified parameters of the Cosserat force source.
 */
void specfem::assembly::sources_impl::compute_source_array(
    const std::shared_ptr<specfem::sources::cosserat_force> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      source->get_x(), source->get_z());
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

      source_array(0, iz, ix) = source->get_f() *
                                std::sin(Kokkos::numbers::pi_v<type_real> /
                                         180 * source->get_angle()) *
                                hlagrange;
      source_array(1, iz, ix) = -1.0 * source->get_f() *
                                std::cos(Kokkos::numbers::pi_v<type_real> /
                                         180 * source->get_angle()) *
                                hlagrange;

      source_array(2, iz, ix) = source->get_fc() * hlagrange;
    }
  }
}

void specfem::assembly::sources_impl::compute_source_array(
    const std::shared_ptr<specfem::sources::external> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      source->get_x(), source->get_z());
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

  // Source array computation
  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      hlagrange = hxi_source(ix) * hgamma_source(iz);

      // Acoustic
      if (el_type == specfem::element::medium_tag::acoustic) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "External source requires 1 component for acoustic medium");
        }
        source_array(0, iz, ix) = hlagrange;
      }
      // Elastic SH
      else if (el_type == specfem::element::medium_tag::elastic_sh) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "External source requires 1 component for elastic SH medium");
        }
        source_array(0, iz, ix) = hlagrange;
      } else if ((el_type == specfem::element::medium_tag::elastic_psv)) {
        if (ncomponents != 2) {
          throw std::runtime_error("External source for elastic PSV, "
                                   "poroelastic, or electromagnetic TE"
                                   "SV requires 2 components");
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
      } else if (el_type == specfem::element::medium_tag::electromagnetic_te) {
        if (ncomponents != 2) {
          throw std::runtime_error(
              "External source requires 2 components for electromagnetic "
              "TE medium");
        }
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = hlagrange;
      } else if (el_type == specfem::element::medium_tag::elastic_psv_t) {
        if (ncomponents != 3) {
          throw std::runtime_error(
              "External source requires 3 components for elastic psv_t medium");
        }
        source_array(0, iz, ix) = hlagrange;
        source_array(1, iz, ix) = hlagrange;
        source_array(2, iz, ix) = hlagrange;
      } else {
        std::ostringstream message;
        message << "Source array computation not implemented for element type: "
                << specfem::element::to_string(el_type) << " [" << __FILE__
                << ":" << __LINE__ << "]";
        auto message_str = message.str();
        Kokkos::abort(message_str.c_str());
      }
    }
  }
}

void specfem::assembly::sources_impl::compute_source_array(
    const std::shared_ptr<specfem::sources::force> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  std::cout << "in compute_array" << std::endl;
  std::cout << "      Force source x: " << source->get_x()
            << ", z: " << source->get_z() << ", angle: " << source->get_angle()
            << std::endl;
  std::cout << "Got vals" << std::endl;
  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      source->get_x(), source->get_z());
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

  // Source array computation
  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      hlagrange = hxi_source(ix) * hgamma_source(iz);

      // Acoustic
      if (el_type == specfem::element::medium_tag::acoustic) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "Force source requires 1 component for acoustic medium");
        }
        source_array(0, iz, ix) = hlagrange;
      }
      // Elastic SH
      else if (el_type == specfem::element::medium_tag::elastic_sh) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "Force source requires 1 component for elastic SH medium");
        }
        source_array(0, iz, ix) = hlagrange;

      } else if ((el_type == specfem::element::medium_tag::elastic_psv)) {
        if (ncomponents != 2) {
          throw std::runtime_error(
              "Force source requires 2 components for elastic, "
              "poroelastic, or electromagnetic-sv media.");
        }
        source_array(0, iz, ix) = std::sin(Kokkos::numbers::pi_v<type_real> /
                                           180 * source->get_angle()) *
                                  hlagrange;
        source_array(1, iz, ix) = -1.0 *
                                  std::cos(Kokkos::numbers::pi_v<type_real> /
                                           180 * source->get_angle()) *
                                  hlagrange;
      } else if ((el_type == specfem::element::medium_tag::poroelastic)) {
        if (ncomponents != 4) {
          throw std::runtime_error(
              "Force source requires 4 components for poroelastic medium");
        }
        source_array(0, iz, ix) = std::sin(Kokkos::numbers::pi_v<type_real> /
                                           180 * source->get_angle()) *
                                  hlagrange;
        source_array(1, iz, ix) = -1.0 *
                                  std::cos(Kokkos::numbers::pi_v<type_real> /
                                           180 * source->get_angle()) *
                                  hlagrange;
        source_array(2, iz, ix) = std::sin(Kokkos::numbers::pi_v<type_real> /
                                           180 * source->get_angle()) *
                                  hlagrange;
        source_array(3, iz, ix) = -1.0 *
                                  std::cos(Kokkos::numbers::pi_v<type_real> /
                                           180 * source->get_angle()) *
                                  hlagrange;
      } else if (el_type == specfem::element::medium_tag::elastic_psv_t) {
        if (ncomponents != 3) {
          throw std::runtime_error(
              "Force source requires 3 components for elastic psv_t medium");
        }
        source_array(0, iz, ix) = std::sin(Kokkos::numbers::pi_v<type_real> /
                                           180 * source->get_angle()) *
                                  hlagrange;
        source_array(1, iz, ix) = -1.0 *
                                  std::cos(Kokkos::numbers::pi_v<type_real> /
                                           180 * source->get_angle()) *
                                  hlagrange;
        source_array(2, iz, ix) = static_cast<type_real>(0.0);
      } else {
        std::ostringstream message;
        message << "Source array computation not implemented for element type: "
                << specfem::element::to_string(el_type) << " [" << __FILE__
                << ":" << __LINE__ << "]";
        auto message_str = message.str();
        Kokkos::abort(message_str.c_str());
      }
    }
  }
}

void specfem::assembly::sources_impl::compute_source_array(
    const std::shared_ptr<specfem::sources::moment_tensor> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      source->get_x(), source->get_z());
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
      source_array(0, iz, ix) =
          source->get_Mxx() * dsrc_dx + source->get_Mxz() * dsrc_dz;
      source_array(1, iz, ix) =
          source->get_Mxz() * dsrc_dx + source->get_Mzz() * dsrc_dz;

      if (el_type == specfem::element::medium_tag::poroelastic) {
        source_array(2, iz, ix) = source_array(0, iz, ix);
        source_array(3, iz, ix) = source_array(1, iz, ix);
      } else if (el_type == specfem::element::medium_tag::elastic_psv_t) {
        source_array(2, iz, ix) = static_cast<type_real>(0.0);
      }
    }
  }
}
