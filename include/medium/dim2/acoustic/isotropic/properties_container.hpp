#pragma once

#include "medium/impl/data_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {
namespace properties {

template <>
struct data_container<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic> {

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  DATA_CONTAINER(rho_inverse, kappa)
};
} // namespace properties

} // namespace medium
} // namespace specfem
