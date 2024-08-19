#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace point {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress_integrand {
  constexpr static int dimension =
      specfem::dimension::dimension<DimensionType>::dim;
  constexpr static int components =
      specfem::medium::medium<DimensionType, MediumTag>::components;

  using simd = specfem::datatype::simd<type_real, UseSIMD>;

  using ViewType = specfem::datatype::VectorPointViewType<type_real, dimension,
                                                          components, UseSIMD>;
  ViewType F;

  KOKKOS_FUNCTION stress_integrand() = default;

  KOKKOS_FUNCTION stress_integrand(const ViewType &F) : F(F) {}
};

} // namespace point
} // namespace specfem
