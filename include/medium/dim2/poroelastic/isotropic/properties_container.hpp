#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace properties {

template <>
struct data_container<specfem::element::medium_tag::poroelastic,
                      specfem::element::property_tag::isotropic> {
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::poroelastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  DATA_CONTAINER(phi, rho_s, rho_f, tortuosity, mu_G, H_Biot, C_Biot, M_Biot,
                 permxx, permxz, permzz, eta_f)
};

} // namespace properties
} // namespace medium
} // namespace specfem
