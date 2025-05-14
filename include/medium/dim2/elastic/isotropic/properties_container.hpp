#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace properties {

template <specfem::element::medium_tag MediumTag>
struct data_container<
    MediumTag, specfem::element::property_tag::isotropic,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  DATA_CONTAINER(lambdaplus2mu, mu, rho)
};

} // namespace properties

} // namespace medium
} // namespace specfem
