#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace kernels {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
struct data_container<
    DimensionTag, MediumTag, specfem::element::property_tag::isotropic,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {
  constexpr static auto dimension = DimensionTag;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  DATA_CONTAINER(rho, mu, kappa, rhop, alpha, beta)
};

} // namespace kernels
} // namespace medium
} // namespace specfem
