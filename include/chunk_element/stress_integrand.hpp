#pragma once

#include "datatypes/chunk_element_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace chunk_element {

template <int NumberElements, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits>
struct stress_integrand {

  constexpr static int num_elements = NumberElements;
  constexpr static int dimension =
      specfem::dimension::dimension<DimensionType>::dim;
  constexpr static int components =
      specfem::medium::medium<DimensionType, MediumTag>::components;

  using ViewType =
      specfem::datatype::VectorChunkViewType<type_real, NumberElements, NGLL,
                                             components, dimension, MemorySpace,
                                             MemoryTraits>;
  ViewType F;

  KOKKOS_FUNCTION stress_integrand() = default;

  KOKKOS_FUNCTION stress_integrand(const ViewType &F) : F(F) {}

  template <typename MemberType>
  KOKKOS_FUNCTION stress_integrand(const MemberType &team)
      : F(team.team_scratch(0)) {}

  static int shmem_size() { return ViewType::shmem_size(); }
};

} // namespace chunk_element
} // namespace specfem
