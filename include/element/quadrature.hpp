#ifndef _ELEMENT_QUADRATURE_HPP
#define _ELEMENT_QUADRATURE_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace element {

namespace impl {

template <typename ViewType, bool StoreHPrimeGLL> struct GLLQuadrature;

template <typename ViewType> struct GLLQuadrature<ViewType, true> {

  using view_type = ViewType;

  ViewType hprime_gll;

  KOKKOS_FUNCTION GLLQuadrature() = default;

  KOKKOS_FUNCTION GLLQuadrature(const ViewType &hprime_gll)
      : hprime_gll(hprime_gll) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION GLLQuadrature(const ScratchMemorySpace &scratch_memory_space)
      : hprime_gll(scratch_memory_space) {}
};

template <typename ViewType> struct GLLQuadrature<ViewType, false> {
  using view_type = ViewType;
};

template <typename ViewType, bool StoreWeightTimesHPrimeGLL>
struct WeightTimesHPrimeGLL;

template <typename ViewType> struct WeightTimesHPrimeGLL<ViewType, true> {
  using view_type = ViewType;
  ViewType hprime_wgll;

  KOKKOS_FUNCTION WeightTimesHPrimeGLL() = default;

  KOKKOS_FUNCTION WeightTimesHPrimeGLL(const ViewType &hprime_wgll)
      : hprime_wgll(hprime_wgll) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION
  WeightTimesHPrimeGLL(const ScratchMemorySpace &scratch_memory_space)
      : hprime_wgll(scratch_memory_space) {}
};

template <typename ViewType> struct WeightTimesHPrimeGLL<ViewType, false> {
  using view_type = ViewType;
};

template <int NGLL, specfem::dimension::type DimensionType,
          typename MemorySpace, typename MemoryTraits, bool StoreHPrimeGLL,
          bool StoreWeightTimesHPrimeGLL>
struct QuadratureTraits
    : public GLLQuadrature<
          Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, MemorySpace,
                       MemoryTraits>,
          StoreHPrimeGLL>,
      public WeightTimesHPrimeGLL<
          Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, MemorySpace,
                       MemoryTraits>,
          StoreWeightTimesHPrimeGLL> {
public:
  constexpr static int ngll = NGLL;
  constexpr static int dimension =
      specfem::dimension::dimension<DimensionType>::value;
  constexpr static bool store_hprime_gll = StoreHPrimeGLL;
  constexpr static bool store_weight_times_hprime_gll =
      StoreWeightTimesHPrimeGLL;

  using memory_space = MemorySpace;

  using ViewType = typename GLLQuadrature<
      Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, MemorySpace,
                   MemoryTraits>,
      StoreHPrimeGLL>::view_type;

private:
  KOKKOS_FUNCTION QuadratureTraits(const ViewType view, std::true_type,
                                   std::false_type)
      : GLLQuadrature<ViewType, StoreHPrimeGLL>(view) {}

  KOKKOS_FUNCTION QuadratureTraits(const ViewType view, std::false_type,
                                   std::true_type)
      : WeightTimesHPrimeGLL<ViewType, StoreWeightTimesHPrimeGLL>(view) {}

  KOKKOS_FUNCTION QuadratureTraits(const ViewType view1, const ViewType view2,
                                   std::true_type, std::true_type)
      : GLLQuadrature<ViewType, StoreHPrimeGLL>(view1),
        WeightTimesHPrimeGLL<ViewType, StoreWeightTimesHPrimeGLL>(view2) {}

  template <typename MemberType>
  KOKKOS_FUNCTION QuadratureTraits(const MemberType &team, std::true_type,
                                   std::false_type)
      : GLLQuadrature<ViewType, StoreHPrimeGLL>(team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION QuadratureTraits(const MemberType &team, std::false_type,
                                   std::true_type)
      : WeightTimesHPrimeGLL<ViewType, StoreWeightTimesHPrimeGLL>(
            team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION QuadratureTraits(const MemberType &team, std::true_type,
                                   std::true_type)
      : GLLQuadrature<ViewType, StoreHPrimeGLL>(team.team_scratch(0)),
        WeightTimesHPrimeGLL<ViewType, StoreWeightTimesHPrimeGLL>(
            team.team_scratch(0)) {}

public:
  KOKKOS_FUNCTION QuadratureTraits() = default;

  KOKKOS_FUNCTION QuadratureTraits(const ViewType &view)
      : QuadratureTraits(
            view, std::integral_constant<bool, StoreHPrimeGLL>{},
            std::integral_constant<bool, StoreWeightTimesHPrimeGLL>{}) {}

  KOKKOS_FUNCTION QuadratureTraits(const ViewType &view1, const ViewType &view2)
      : QuadratureTraits(
            view1, view2, std::integral_constant<bool, StoreHPrimeGLL>{},
            std::integral_constant<bool, StoreWeightTimesHPrimeGLL>{}) {}

  template <typename MemberType>
  KOKKOS_FUNCTION QuadratureTraits(const MemberType &team)
      : QuadratureTraits(
            team, std::integral_constant<bool, StoreHPrimeGLL>{},
            std::integral_constant<bool, StoreWeightTimesHPrimeGLL>{}) {}

  static int shmem_size() {
    return (static_cast<int>(StoreHPrimeGLL) +
            static_cast<int>(StoreWeightTimesHPrimeGLL)) *
           ViewType::shmem_size();
  }
};
} // namespace impl

template <int NGLL, specfem::dimension::type DimensionType,
          typename MemorySpace, typename MemoryTraits, bool StoreHPrimeGLL,
          bool WeightTimesDerivatives>
struct quadrature
    : public impl::QuadratureTraits<NGLL, DimensionType, MemorySpace,
                                    MemoryTraits, StoreHPrimeGLL,
                                    WeightTimesDerivatives> {

  using ViewType =
      typename impl::QuadratureTraits<NGLL, DimensionType, MemorySpace,
                                      MemoryTraits, StoreHPrimeGLL,
                                      WeightTimesDerivatives>::ViewType;

  KOKKOS_FUNCTION quadrature() = default;

  KOKKOS_FUNCTION quadrature(const ViewType &view)
      : impl::QuadratureTraits<NGLL, DimensionType, MemorySpace, MemoryTraits,
                               StoreHPrimeGLL, WeightTimesDerivatives>(view) {}

  KOKKOS_FUNCTION quadrature(const ViewType &view1, const ViewType &view2)
      : impl::QuadratureTraits<NGLL, DimensionType, MemorySpace, MemoryTraits,
                               StoreHPrimeGLL, WeightTimesDerivatives>(view1,
                                                                       view2) {}

  template <typename MemberType>
  KOKKOS_FUNCTION quadrature(const MemberType &team)
      : impl::QuadratureTraits<NGLL, DimensionType, MemorySpace, MemoryTraits,
                               StoreHPrimeGLL, WeightTimesDerivatives>(team) {
    static_assert(
        Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                   MemorySpace>::accessible,
        "MemorySpace is not accessible from the execution space");
  }
};

// template <int NGLL, typename MemorySpace, typename MemoryTraits>
// struct quadrature<NGLL, specfem::dimension::type::dim2, MemorySpace,
//                   MemoryTraits, true, true>
//     : public impl::QuadratureTraits<NGLL, specfem::dimension::type::dim2,
//                                     MemorySpace, MemoryTraits, true, true> {

//   using ViewType =
//       typename impl::QuadratureTraits<NGLL, specfem::dimension::type::dim2,
//                                       MemorySpace, MemoryTraits, true,
//                                       true>::ViewType;
//   ViewType hprime_gll;
//   ViewType hprimew_gll;

//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION quadrature(const MemberType &team)
//       : hprime_gll(team.team_scratch(0)), hprimew_gll(team.team_scratch(0))
//       {}

//   KOKKOS_FUNCTION quadrature(const ViewType &hprime_gll,
//                              const ViewType &hprimew_gll)
//       : hprime_gll(hprime_gll), hprimew_gll(hprimew_gll) {}

//   static int shmem_size() { return 2 * ViewType::shmem_size(); }
// };

// template <int NGLL, typename MemorySpace, typename MemoryTraits>
// struct quadrature<NGLL, specfem::dimension::type::dim2, MemorySpace,
//                   MemoryTraits, true, false>
//     : public impl::QuadratureTraits<NGLL, specfem::dimension::type::dim2,
//                                     MemorySpace, MemoryTraits, true, false> {

//   using ViewType =
//       typename impl::QuadratureTraits<NGLL, specfem::dimension::type::dim2,
//                                       MemorySpace, MemoryTraits, true,
//                                       false>::ViewType;
//   ViewType hprime_gll;

//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION quadrature(const MemberType &team)
//       : hprime_gll(team.team_scratch(0)) {}

//   KOKKOS_FUNCTION quadrature(const ViewType &hprime_gll)
//       : hprime_gll(hprime_gll) {}

//   static int shmem_size() { return ViewType::shmem_size(); }
// };

} // namespace element
} // namespace specfem

#endif
