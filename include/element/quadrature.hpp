#ifndef _ELEMENT_QUADRATURE_HPP
#define _ELEMENT_QUADRATURE_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace element {
template <int NGLL, specfem::dimension::type DimensionType,
          typename MemorySpace, typename MemoryTraits,
          bool StoreGLLQuadratureDerivatives, bool WeightTimesDerivatives>
struct quadrature;

template <int NGLL, typename MemorySpace, typename MemoryTraits>
struct quadrature<NGLL, specfem::dimension::type::dim2, MemorySpace,
                  MemoryTraits, true, true> {

  using ViewType = Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
                                MemorySpace, MemoryTraits>;

  ViewType hprime_gll;
  ViewType hprimew_gll;

  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  quadrature(const MemberType &team)
      : hprime_gll(team.team_scratch(0)), hprimew_gll(team.team_scratch(0)) {}

  quadrature(const ViewType &hprime_gll, const ViewType &hprimew_gll)
      : hprime_gll(hprime_gll), hprimew_gll(hprimew_gll) {}

  static int shmem_size() { return 2 * ViewType::shmem_size(); }
};

template <int NGLL, typename MemorySpace, typename MemoryTraits>
struct quadrature<NGLL, specfem::dimension::type::dim2, MemorySpace,
                  MemoryTraits, true, false> {

  using ViewType = Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
                                MemorySpace, MemoryTraits>;

  ViewType hprime_gll;

  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  quadrature(const MemberType &team) : hprime_gll(team.team_scratch(0)) {}

  quadrature(const ViewType &hprime_gll) : hprime_gll(hprime_gll) {}

  static int shmem_size() { return ViewType::shmem_size(); }
};

} // namespace element
} // namespace specfem

#endif
