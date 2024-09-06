#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace element {

namespace impl {

template <typename ViewType, bool StoreHPrimeGLL> struct GLLQuadrature {

  using view_type = ViewType;

  ViewType hprime_gll; ///< Derivatives of lagrange polynomials \f$l'\f$ at GLL
                       ///< points. Defined if StoreHPrimeGLL is true.

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
struct WeightTimesHPrimeGLL {
  using view_type = ViewType;
  ViewType hprime_wgll; ///< Weight times derivatives of lagrange polynomials
                        ///< \f$ w_j l'_{i,j} \f$ at GLL points. Defined if
                        ///< StoreWeightTimesHPrimeGLL is true.

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

template <typename ViewType, bool StoreHPrimeGLL,
          bool StoreWeightTimesHPrimeGLL>
struct ImplQuadratureTraits
    : public GLLQuadrature<ViewType, StoreHPrimeGLL>,
      public WeightTimesHPrimeGLL<ViewType, StoreWeightTimesHPrimeGLL> {
public:
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static bool store_hprime_gll =
      StoreHPrimeGLL; ///< Boolean to indicate if derivatives of Lagrange
                      ///< polynomials are stored
  constexpr static bool store_weight_times_hprime_gll =
      StoreWeightTimesHPrimeGLL; ///< Boolean to indicate if weight times
                                 ///< derivatives of Lagrange polynomials are
                                 ///< stored

  using view_type = ViewType;
  ///@}

private:
  KOKKOS_FUNCTION ImplQuadratureTraits(const ViewType view, std::true_type,
                                       std::false_type)
      : GLLQuadrature<ViewType, StoreHPrimeGLL>(view) {}

  KOKKOS_FUNCTION ImplQuadratureTraits(const ViewType view, std::false_type,
                                       std::true_type)
      : WeightTimesHPrimeGLL<ViewType, StoreWeightTimesHPrimeGLL>(view) {}

  KOKKOS_FUNCTION ImplQuadratureTraits(const ViewType view1,
                                       const ViewType view2, std::true_type,
                                       std::true_type)
      : GLLQuadrature<ViewType, StoreHPrimeGLL>(view1),
        WeightTimesHPrimeGLL<ViewType, StoreWeightTimesHPrimeGLL>(view2) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplQuadratureTraits(const MemberType &team, std::true_type,
                                       std::false_type)
      : GLLQuadrature<ViewType, StoreHPrimeGLL>(team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplQuadratureTraits(const MemberType &team, std::false_type,
                                       std::true_type)
      : WeightTimesHPrimeGLL<ViewType, StoreWeightTimesHPrimeGLL>(
            team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplQuadratureTraits(const MemberType &team, std::true_type,
                                       std::true_type)
      : GLLQuadrature<ViewType, StoreHPrimeGLL>(team.team_scratch(0)),
        WeightTimesHPrimeGLL<ViewType, StoreWeightTimesHPrimeGLL>(
            team.team_scratch(0)) {}

public:
  KOKKOS_FUNCTION ImplQuadratureTraits() = default;

  KOKKOS_FUNCTION ImplQuadratureTraits(const ViewType &view)
      : ImplQuadratureTraits(
            view, std::integral_constant<bool, StoreHPrimeGLL>{},
            std::integral_constant<bool, StoreWeightTimesHPrimeGLL>{}) {}

  KOKKOS_FUNCTION ImplQuadratureTraits(const ViewType &view1,
                                       const ViewType &view2)
      : ImplQuadratureTraits(
            view1, view2, std::integral_constant<bool, StoreHPrimeGLL>{},
            std::integral_constant<bool, StoreWeightTimesHPrimeGLL>{}) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplQuadratureTraits(const MemberType &team)
      : ImplQuadratureTraits(
            team, std::integral_constant<bool, StoreHPrimeGLL>{},
            std::integral_constant<bool, StoreWeightTimesHPrimeGLL>{}) {}

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int Amount of shared memory in bytes
   */
  constexpr static int shmem_size() {
    return (static_cast<int>(StoreHPrimeGLL) +
            static_cast<int>(StoreWeightTimesHPrimeGLL)) *
           ViewType::shmem_size();
  }
};

template <int NGLL, specfem::dimension::type DimensionType,
          typename MemorySpace, typename MemoryTraits, bool StoreHPrimeGLL,
          bool StoreWeightTimesHPrimeGLL>
struct QuadratureTraits
    : public ImplQuadratureTraits<
          Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight, MemorySpace,
                       MemoryTraits>,
          StoreHPrimeGLL, StoreWeightTimesHPrimeGLL> {

  constexpr static int ngll = NGLL;
  constexpr static int dimension =
      specfem::dimension::dimension<DimensionType>::dim;
  using ViewType = Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
                                MemorySpace, MemoryTraits>;

  KOKKOS_FUNCTION QuadratureTraits() = default;

  KOKKOS_FUNCTION QuadratureTraits(const ViewType &view)
      : ImplQuadratureTraits<ViewType, StoreHPrimeGLL,
                             StoreWeightTimesHPrimeGLL>(view) {}

  KOKKOS_FUNCTION QuadratureTraits(const ViewType &view1, const ViewType &view2)
      : ImplQuadratureTraits<ViewType, StoreHPrimeGLL,
                             StoreWeightTimesHPrimeGLL>(view1, view2) {}

  template <typename MemberType>
  KOKKOS_FUNCTION QuadratureTraits(const MemberType &team)
      : ImplQuadratureTraits<ViewType, StoreHPrimeGLL,
                             StoreWeightTimesHPrimeGLL>(team) {
    static_assert(
        Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                   MemorySpace>::accessible,
        "MemorySpace is not accessible from the execution space");
  }
};

} // namespace impl

/**
 * @brief Struct used to store quadrature values within an element.
 *
 * Currently we store the derivatives of the Lagrange polynomials and the weight
 * times the derivatives of the Lagrange polynomials. Since these are the
 * variables required when computing gradients and divergences required to
 * compute forces.
 *
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points
 * @tparam DimensionType Dimension of the element
 * @tparam MemorySpace Memory space for the views
 * @tparam MemoryTraits Memory traits for the views
 * @tparam StoreHPrimeGLL Whether to store the derivatives of the Lagrange
 * polynomials
 * @tparam WeightTimesDerivatives Whether to store the weight times the
 * derivatives of the Lagrange polynomials
 */
template <int NGLL, specfem::dimension::type DimensionType,
          typename MemorySpace, typename MemoryTraits, bool StoreHPrimeGLL,
          bool StoreWeightTimesHPrimeGLL>
struct quadrature
    : public impl::QuadratureTraits<NGLL, DimensionType, MemorySpace,
                                    MemoryTraits, StoreHPrimeGLL,
                                    StoreWeightTimesHPrimeGLL> {

  /**
   * @name Typedefs
   *
   */
  ///@{

  /**
   * @brief Underlying view type used to store quadrature values.
   *
   */
  using ViewType =
      typename impl::QuadratureTraits<NGLL, DimensionType, MemorySpace,
                                      MemoryTraits, StoreHPrimeGLL,
                                      StoreWeightTimesHPrimeGLL>::ViewType;
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION quadrature() = default;

  /**
   * @brief Constructor that initializes the quadrature with a view.
   *
   * Enabled if only one of StoreHPrimeGLL or WeightTimesDerivatives is true.
   *
   * @param view View to initialize the quadrature with. Either equal to @c
   * hprime_gll or @c hprime_wgll.
   */
  KOKKOS_FUNCTION quadrature(const ViewType &view)
      : impl::QuadratureTraits<NGLL, DimensionType, MemorySpace, MemoryTraits,
                               StoreHPrimeGLL, StoreWeightTimesHPrimeGLL>(
            view) {}

  /**
   * @brief Constructor that initializes the quadrature with two views.
   *
   * Enabled if both StoreHPrimeGLL and WeightTimesDerivatives are true.
   *
   * @param view1 View to initialize the quadrature with. Equal to @c
   * hprime_gll.
   * @param view2 View to initialize the quadrature with. Equal to @c
   * hprime_wgll.
   */
  KOKKOS_FUNCTION quadrature(const ViewType &view1, const ViewType &view2)
      : impl::QuadratureTraits<NGLL, DimensionType, MemorySpace, MemoryTraits,
                               StoreHPrimeGLL, StoreWeightTimesHPrimeGLL>(
            view1, view2) {}

  /**
   * @brief Constructor that initializes the quadrature within Scratch Memory.
   *
   * @tparam MemberType Type of the Kokkos team member.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_FUNCTION quadrature(const MemberType &team)
      : impl::QuadratureTraits<NGLL, DimensionType, MemorySpace, MemoryTraits,
                               StoreHPrimeGLL, StoreWeightTimesHPrimeGLL>(
            team) {
    static_assert(
        Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                   MemorySpace>::accessible,
        "MemorySpace is not accessible from the execution space");
  }
  ///@}
};

} // namespace element
} // namespace specfem
