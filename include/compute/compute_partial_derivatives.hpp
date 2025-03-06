#ifndef _COMPUTE_PARTIAL_DERIVATIVES_HPP
#define _COMPUTE_PARTIAL_DERIVATIVES_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "point/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
/**
 * @brief Partial derivatives of the basis functions at every quadrature point
 *
 */
struct partial_derivatives {

private:
  using ViewType =
      typename Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                            Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                            ///< type used to
                                                            ///< store data

public:
  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z direction
  int ngllx; ///< Number of quadrature points in x direction

  ViewType xix;                    ///< @xix
  ViewType::HostMirror h_xix;      ///< Host mirror of @xix
  ViewType xiz;                    ///< @xiz
  ViewType::HostMirror h_xiz;      ///< Host mirror of @xiz
  ViewType gammax;                 ///< @gammax
  ViewType::HostMirror h_gammax;   ///< Host mirror of @gammax
  ViewType gammaz;                 ///< @gammaz
  ViewType::HostMirror h_gammaz;   ///< Host mirror of @gammaz
  ViewType jacobian;               ///< Jacobian
  ViewType::HostMirror h_jacobian; ///< Host mirror of Jacobian

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  partial_derivatives() = default;

  partial_derivatives(const int nspec, const int ngllz, const int ngllx);
  /**
   * @brief Construct a new partial derivatives object from mesh information
   *
   * @param mesh Mesh information
   */
  partial_derivatives(const specfem::compute::mesh &mesh);
  ///@}

  void sync_views();
};

/**
 * @defgroup ComputePartialDerivativesDataAccess
 *
 * @brief Functions to load and store partial derivatives at a given quadrature
 * point
 *
 */

template <bool on_device, typename PointPartialDerivativesType,
          typename std::enable_if_t<
              PointPartialDerivativesType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(
    const specfem::point::simd_index<PointPartialDerivativesType::dimension>
        &index,
    const specfem::compute::partial_derivatives &derivatives,
    PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int nspec = derivatives.nspec;
  const int iz = index.iz;
  const int ix = index.ix;

  using simd = typename PointPartialDerivativesType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  if constexpr (on_device) {
    Kokkos::Experimental::where(mask, partial_derivatives.xix)
        .copy_from(&derivatives.xix(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.gammax)
        .copy_from(&derivatives.gammax(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.xiz)
        .copy_from(&derivatives.xiz(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.gammaz)
        .copy_from(&derivatives.gammaz(ispec, iz, ix), tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, partial_derivatives.jacobian)
          .copy_from(&derivatives.jacobian(ispec, iz, ix), tag_type());
    }
  } else {
    Kokkos::Experimental::where(mask, partial_derivatives.xix)
        .copy_from(&derivatives.h_xix(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.gammax)
        .copy_from(&derivatives.h_gammax(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.xiz)
        .copy_from(&derivatives.h_xiz(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.gammaz)
        .copy_from(&derivatives.h_gammaz(ispec, iz, ix), tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, partial_derivatives.jacobian)
          .copy_from(&derivatives.h_jacobian(ispec, iz, ix), tag_type());
    }
  }
}

template <bool on_device, typename PointPartialDerivativesType,
          typename std::enable_if_t<
              !PointPartialDerivativesType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(
    const specfem::point::index<PointPartialDerivativesType::dimension> &index,
    const specfem::compute::partial_derivatives &derivatives,
    PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  if constexpr (on_device) {
    partial_derivatives.xix = derivatives.xix(ispec, iz, ix);
    partial_derivatives.gammax = derivatives.gammax(ispec, iz, ix);
    partial_derivatives.xiz = derivatives.xiz(ispec, iz, ix);
    partial_derivatives.gammaz = derivatives.gammaz(ispec, iz, ix);
    if constexpr (StoreJacobian) {
      partial_derivatives.jacobian = derivatives.jacobian(ispec, iz, ix);
    }
  } else {
    partial_derivatives.xix = derivatives.h_xix(ispec, iz, ix);
    partial_derivatives.gammax = derivatives.h_gammax(ispec, iz, ix);
    partial_derivatives.xiz = derivatives.h_xiz(ispec, iz, ix);
    partial_derivatives.gammaz = derivatives.h_gammaz(ispec, iz, ix);
    if constexpr (StoreJacobian) {
      partial_derivatives.jacobian = derivatives.h_jacobian(ispec, iz, ix);
    }
  }
}

template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              PointPartialDerivativesType::simd::using_simd, int> = 0>
inline void impl_store_on_host(
    const specfem::point::simd_index<PointPartialDerivativesType::dimension>
        &index,
    const specfem::compute::partial_derivatives &derivatives,
    const PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int nspec = derivatives.nspec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  using simd = typename PointPartialDerivativesType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  Kokkos::Experimental::where(mask, partial_derivatives.xix)
      .copy_to(&derivatives.h_xix(ispec, iz, ix), tag_type());
  Kokkos::Experimental::where(mask, partial_derivatives.gammax)
      .copy_to(&derivatives.h_gammax(ispec, iz, ix), tag_type());
  Kokkos::Experimental::where(mask, partial_derivatives.xiz)
      .copy_to(&derivatives.h_xiz(ispec, iz, ix), tag_type());
  Kokkos::Experimental::where(mask, partial_derivatives.gammaz)
      .copy_to(&derivatives.h_gammaz(ispec, iz, ix), tag_type());
  if constexpr (StoreJacobian) {
    Kokkos::Experimental::where(mask, partial_derivatives.jacobian)
        .copy_to(&derivatives.h_jacobian(ispec, iz, ix), tag_type());
  }
}

template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              !PointPartialDerivativesType::simd::using_simd, int> = 0>
inline void impl_store_on_host(
    const specfem::point::index<PointPartialDerivativesType::dimension> &index,
    const specfem::compute::partial_derivatives &derivatives,
    const PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  derivatives.h_xix(ispec, iz, ix) = partial_derivatives.xix;
  derivatives.h_gammax(ispec, iz, ix) = partial_derivatives.gammax;
  derivatives.h_xiz(ispec, iz, ix) = partial_derivatives.xiz;
  derivatives.h_gammaz(ispec, iz, ix) = partial_derivatives.gammaz;
  if constexpr (StoreJacobian) {
    derivatives.h_jacobian(ispec, iz, ix) = partial_derivatives.jacobian;
  }
}

/**
 * @brief Load the partial derivatives at a given quadrature point on the device
 *
 * @ingroup ComputePartialDerivativesDataAccess
 *
 * @tparam PointPartialDerivativesType Point partial derivatives type. Needs to
 * be of @ref specfem::point::partial_derivatives
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param derivatives Partial derivatives container
 * @param partial_derivatives Partial derivatives at the given quadrature point
 */
template <
    typename PointPartialDerivativesType, typename IndexType,
    typename std::enable_if_t<IndexType::using_simd ==
                                  PointPartialDerivativesType::simd::using_simd,
                              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const IndexType &index,
               const specfem::compute::partial_derivatives &derivatives,
               PointPartialDerivativesType &partial_derivatives) {
  impl_load<true>(index, derivatives, partial_derivatives);
}

/**
 * @brief Store the partial derivatives at a given quadrature point on the
 * device
 *
 * @ingroup ComputePartialDerivativesDataAccess
 *
 * @tparam PointPartialDerivativesType Point partial derivatives type. Needs to
 * be of @ref specfem::point::partial_derivatives
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param derivatives Partial derivatives container
 * @param partial_derivatives Partial derivatives at the given quadrature point
 */
template <
    typename PointPartialDerivativesType, typename IndexType,
    typename std::enable_if_t<IndexType::using_simd ==
                                  PointPartialDerivativesType::simd::using_simd,
                              int> = 0>
inline void
load_on_host(const IndexType &index,
             const specfem::compute::partial_derivatives &derivatives,
             PointPartialDerivativesType &partial_derivatives) {
  impl_load<false>(index, derivatives, partial_derivatives);
}

/**
 * @brief Store the partial derivatives at a given quadrature point on the
 * device
 *
 * @ingroup ComputePartialDerivativesDataAccess
 *
 * @tparam PointPartialDerivativesType Point partial derivatives type. Needs to
 * be of @ref specfem::point::partial_derivatives
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param derivatives Partial derivatives container
 * @param partial_derivatives Partial derivatives at the given quadrature point
 */
template <
    typename PointPartialDerivativesType, typename IndexType,
    typename std::enable_if_t<IndexType::using_simd ==
                                  PointPartialDerivativesType::simd::using_simd,
                              int> = 0>
inline void
store_on_host(const IndexType &index,
              const specfem::compute::partial_derivatives &derivatives,
              const PointPartialDerivativesType &partial_derivatives) {
  impl_store_on_host(index, derivatives, partial_derivatives);
}
} // namespace compute
} // namespace specfem

#endif
