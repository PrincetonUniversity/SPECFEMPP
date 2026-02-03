#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/datatype.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace quadrature {
/**
 * @brief Struct used to store derivatives of Lagrange interpolants within an
 * element at GLL points.
 *
 * Currently we store the derivatives of the Lagrange polynomials since these
 * are the variables required when computing gradients and divergences in
 * compute_forces.
 *
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points
 * @tparam DimensionTag Dimension of the element
 * @tparam MemorySpace Memory space for the views
 * @tparam MemoryTraits Memory traits for the views
 */
template <int NGLL, specfem::dimension::type DimensionTag, typename MemorySpace,
          typename MemoryTraits>
struct lagrange_derivative {
  /**
   * @name Typedefs
   *
   */
  ///@{

  /**
   * @brief Underlying view type used to store lagrange_derivative values.
   *
   */
  using ViewType = Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
                                MemorySpace, MemoryTraits>;
  ViewType hprime_gll; ///< Derivatives of lagrange polynomials \f$l'\f$ at GLL
                       ///< points.
  constexpr static auto dimension_tag =
      DimensionTag;                 ///< Dimension tag (dim2 or dim3)
  constexpr static int ngll = NGLL; ///< Number of GLL points
  constexpr static auto data_class_type =
      specfem::data_access::DataClassType::lagrange_derivative; ///< Data class
                                                                ///< type
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
  KOKKOS_FUNCTION lagrange_derivative() = default;

  /**
   * @brief Constructor that initializes the lagrange_derivative with a view.
   *
   * @param view View to initialize the lagrange_derivative with.
   */
  KOKKOS_FUNCTION lagrange_derivative(const ViewType &view)
      : hprime_gll(view) {}

  /**
   * @brief Constructor that initializes the lagrange_derivative within Scratch
   * Memory.
   *
   * @tparam MemberType Type of the Kokkos team member.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_FUNCTION lagrange_derivative(const MemberType &team)
      : hprime_gll(team.team_scratch(0)) {
    static_assert(
        Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                   MemorySpace>::accessible,
        "MemorySpace is not accessible from the execution space");
  }
  ///@}
  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int Amount of shared memory in bytes
   */
  constexpr static int shmem_size() { return ViewType::shmem_size(); }

  /**
   * @brief Access the derivative of the Lagrange polynomial at GLL points
   *
   * @param l Index of the Lagrange polynomial
   * @param ix Index of the GLL point
   * @return Reference to the derivative value
   */
  KOKKOS_INLINE_FUNCTION constexpr const auto &xi(const int l,
                                                  const int ix) const {
    return hprime_gll(l, ix);
  }

  /**
   * @brief Access the derivative of the Lagrange polynomial at GLL points
   *
   * @param l Index of the Lagrange polynomial
   * @param iy Index of the GLL point
   * @return Reference to the derivative value
   */
  template <
      specfem::dimension::type D = DimensionTag,
      typename std::enable_if_t<D == specfem::dimension::type::dim3, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr const auto &eta(const int l,
                                                   const int iy) const {
    return hprime_gll(l, iy);
  }

  /**
   * @brief Access the derivative of the Lagrange polynomial at GLL points
   *
   * @param l Index of the Lagrange polynomial
   * @param iz Index of the GLL point
   * @return Reference to the derivative value
   */
  KOKKOS_INLINE_FUNCTION constexpr const auto &gamma(const int l,
                                                     const int iz) const {
    return hprime_gll(l, iz);
  }
};

} // namespace quadrature
} // namespace specfem
