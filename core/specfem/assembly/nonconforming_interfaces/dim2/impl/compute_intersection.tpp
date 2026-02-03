#pragma once

#include "Kokkos_Core_fwd.hpp"
#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "compute_intersection.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem/jacobian/dim2/jacobian.hpp"
#include "specfem/point/global_coordinates.hpp"
#include "specfem_setup.hpp"
#include <sstream>
#include <stdexcept>

template <typename TransferView1, typename TransferView2,
          typename TransferView3, typename TransferView4>
void specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element1,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element2,
    const specfem::mesh_entity::dim2::type &edge1,
    const specfem::mesh_entity::dim2::type &edge2,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &element_quadrature,
    TransferView1 &transfer_function1, TransferView2 &transfer_function1_prime,
    TransferView3 &transfer_function2,
    TransferView4 &transfer_function2_prime) {

  // ======= ensure array shapes are correct =======
  const auto nquad_mortar = mortar_quadrature.extent(0);
  const auto ngll = element_quadrature.extent(0);
  if ((transfer_function1.extent(0) != nquad_mortar ||
       transfer_function1_prime.extent(0) != nquad_mortar ||
       transfer_function1.extent(1) != ngll ||
       transfer_function1_prime.extent(1) != ngll) ||
      (transfer_function2.extent(0) != nquad_mortar ||
       transfer_function2_prime.extent(0) != nquad_mortar ||
       transfer_function2.extent(1) != ngll ||
       transfer_function2_prime.extent(1) != ngll)) {
    std::ostringstream oss;
    oss << "Incompatible dimensions of `Kokkos::View`s in "
           "specfem::assembly::nonconforming_interfaces_impl:"
           ":set_transfer_functions:\n";
    if (transfer_function1.extent(0) != nquad_mortar ||
        transfer_function1_prime.extent(0) != nquad_mortar ||
        transfer_function2.extent(0) != nquad_mortar ||
        transfer_function2_prime.extent(0) != nquad_mortar) {
      oss << "Mortar quadrature has " << nquad_mortar
          << " quadrature points, which should match the first axis of the "
             "transfer function tensor.";
    }
    if (transfer_function1.extent(1) != ngll ||
        transfer_function1_prime.extent(1) != ngll ||
        transfer_function2.extent(1) != ngll ||
        transfer_function2_prime.extent(1) != ngll) {
      oss << "Edge quadrature (element ngll) has " << ngll
          << " quadrature points, which should match the second axis of the "
             "transfer function tensor.";
    }
    oss << "\n Shape of\n"
        << "        transfer_function1: (" << transfer_function1.extent(0)
        << ", " << transfer_function1.extent(1) << ")\n"
        << "  transfer_function1_prime: (" << transfer_function1_prime.extent(0)
        << ", " << transfer_function1_prime.extent(1) << ")\n"
        << "        transfer_function2: (" << transfer_function2.extent(0)
        << ", " << transfer_function2.extent(1) << ")\n"
        << "  transfer_function2_prime: (" << transfer_function2_prime.extent(0)
        << ", " << transfer_function2_prime.extent(1) << ")\n";
    throw std::runtime_error(oss.str());
  }

  // ======= populate transfer function and deriv =======

  const auto intersections =
      specfem::assembly::nonconforming_interfaces_impl::compute_intersection(
          element1, element2, edge1, edge2, mortar_quadrature);

  for (int iquad = 0; iquad < nquad_mortar; iquad++) {
    {
      const auto [hxi, hpxi] =
          specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
              intersections[iquad].first, ngll, element_quadrature);
      for (int igll = 0; igll < ngll; igll++) {
        transfer_function1(iquad, igll) = hxi(igll);
        transfer_function1_prime(iquad, igll) = hpxi(igll);
      }
    }
    {
      const auto [hxi, hpxi] =
          specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
              intersections[iquad].second, ngll, element_quadrature);
      for (int igll = 0; igll < ngll; igll++) {
        transfer_function2(iquad, igll) = hxi(igll);
        transfer_function2_prime(iquad, igll) = hpxi(igll);
      }
    }
  }
}

template <typename TransferView1, typename TransferView2>
void specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element1,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element2,
    const specfem::mesh_entity::dim2::type &edge1,
    const specfem::mesh_entity::dim2::type &edge2,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &element_quadrature,
    TransferView1 &transfer_function1, TransferView2 &transfer_function2) {
  const auto nquad_mortar = mortar_quadrature.extent(0);
  const auto ngll = element_quadrature.extent(0);
  Kokkos::View<type_real **, Kokkos::HostSpace> devnull("prime_capture",
                                                        nquad_mortar, ngll);
  set_transfer_functions(element1, element2, edge1, edge2, mortar_quadrature,
                         element_quadrature, transfer_function1, devnull,
                         transfer_function2, devnull);
}
