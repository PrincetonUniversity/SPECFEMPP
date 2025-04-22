#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace kernels {
template <>
struct data_container<specfem::element::medium_tag::poroelastic,
                      specfem::element::property_tag::isotropic> {
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::poroelastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  DATA_CONTAINER(rhot, rhof, eta, sm, mu_fr, B, C, M, mu_frb, rhob, rhofb, phi,
                 cpI, cpII, cs, rhobb, rhofbb, ratio, phib);
};
} // namespace kernels

} // namespace medium
} // namespace specfem
