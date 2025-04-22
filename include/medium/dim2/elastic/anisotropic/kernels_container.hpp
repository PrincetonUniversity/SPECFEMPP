#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace kernels {

template <specfem::element::medium_tag MediumTag>
struct data_container<
    MediumTag, specfem::element::property_tag::anisotropic,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;

  DATA_CONTAINER(rho, c11, c13, c15, c33, c35, c55)
};

} // namespace kernels

} // namespace medium
} // namespace specfem
