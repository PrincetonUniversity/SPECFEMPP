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
 * @brief Partial derivates matrices required to compute integrals
 *
 * The matrices are stored in (ispec, iz, ix) format
 *
 */
struct partial_derivatives {

  using ViewType = typename Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                         Kokkos::DefaultExecutionSpace>;

  int nspec;                     ///< Number of spectral elements
  int ngllz;                     ///< Number of quadrature points in z direction
  int ngllx;                     ///< Number of quadrature points in x direction
  ViewType xix;                  ///< inverted partial derivates
                                 ///< @xix stored on the device
  ViewType::HostMirror h_xix;    ///< inverted partial
                                 ///< derivates @xix stored on
                                 ///< the host
  ViewType xiz;                  ///< inverted partial derivates
                                 ///< @xiz stored on the device
  ViewType::HostMirror h_xiz;    ///< inverted partial
                                 ///< derivates @xiz stored on
                                 ///< the host
  ViewType gammax;               ///< inverted partial
                                 ///< derivates @gammax
                                 ///< stored on device
  ViewType::HostMirror h_gammax; ///< inverted partial
                                 ///< derivates @gammax
                                 ///< stored on host
  ViewType gammaz;               ///< inverted partial
                                 ///< derivates @gammaz
                                 ///< stored on device
  ViewType::HostMirror h_gammaz; ///< inverted partial
                                 ///< derivates @gammaz
                                 ///< stored on host
  ViewType jacobian;             ///< Jacobian values stored
                                 ///< on device
  ViewType::HostMirror h_jacobian; ///< Jacobian values
                                   ///< stored on host
  /**
   * @brief Default constructor
   *
   */
  partial_derivatives() = default;
  /**
   * @brief Constructor to allocate views
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of quadrature points in z direction
   * @param ngllx Number of quadrature points in x direction
   */
  partial_derivatives(const int nspec, const int ngllz, const int ngllx);
  /**
   * @brief Constructor to allocate and assign views
   *
   * @param coorg (x,z) for every spectral element control node
   * @param knods Global control element number for every control node
   * @param quadx Quadrature object in x dimension
   * @param quadz Quadrature object in z dimension
   */
  partial_derivatives(const specfem::compute::mesh &mesh);
  /**
   * @brief Helper routine to sync views within this struct
   *
   */
  void sync_views();

  // template <bool load_jacobian>
  // KOKKOS_INLINE_FUNCTION specfem::point::partial_derivatives2
  // load_device_derivatives(const int ispec, const int iz, const int ix) const
  // {
  //   if constexpr (load_jacobian) {
  //     return { xix(ispec, iz, ix), gammax(ispec, iz, ix), xiz(ispec, iz, ix),
  //              gammaz(ispec, iz, ix), jacobian(ispec, iz, ix) };
  //   } else {
  //     return { xix(ispec, iz, ix), gammax(ispec, iz, ix), xiz(ispec, iz, ix),
  //              gammaz(ispec, iz, ix) };
  //   }
  // };

  // template <bool load_jacobian>
  // specfem::point::partial_derivatives2
  // load_host_derivatives(const int ispec, const int iz, const int ix) const {
  //   if constexpr (load_jacobian) {
  //     return { h_xix(ispec, iz, ix), h_gammax(ispec, iz, ix),
  //              h_xiz(ispec, iz, ix), h_gammaz(ispec, iz, ix),
  //              h_jacobian(ispec, iz, ix) };
  //   } else {
  //     return { h_xix(ispec, iz, ix), h_gammax(ispec, iz, ix),
  //              h_xiz(ispec, iz, ix), h_gammaz(ispec, iz, ix) };
  //   }
  // };
};

template <typename PartialDerivativesType,
          typename std::enable_if_t<PartialDerivativesType::simd::using_simd,
                                    int> = 0>
NOINLINE KOKKOS_FUNCTION void load_on_device(
    const specfem::point::simd_index<PartialDerivativesType::dimension> &index,
    const specfem::compute::partial_derivatives &derivatives,
    PartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int nspec = derivatives.nspec;
  const int iz = index.iz;
  const int ix = index.ix;

  using simd = typename PartialDerivativesType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  constexpr static bool StoreJacobian = PartialDerivativesType::store_jacobian;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

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
}

template <typename PartialDerivativesType,
          typename std::enable_if_t<!PartialDerivativesType::simd::using_simd,
                                    int> = 0>
NOINLINE KOKKOS_FUNCTION void load_on_device(
    const specfem::point::index<PartialDerivativesType::dimension> &index,
    const specfem::compute::partial_derivatives &derivatives,
    PartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PartialDerivativesType::store_jacobian;

  partial_derivatives.xix = derivatives.xix(ispec, iz, ix);
  partial_derivatives.gammax = derivatives.gammax(ispec, iz, ix);
  partial_derivatives.xiz = derivatives.xiz(ispec, iz, ix);
  partial_derivatives.gammaz = derivatives.gammaz(ispec, iz, ix);
  if constexpr (StoreJacobian) {
    partial_derivatives.jacobian = derivatives.jacobian(ispec, iz, ix);
  }
}

template <typename PartialDerivativesType,
          typename std::enable_if_t<PartialDerivativesType::simd::using_simd,
                                    int> = 0>
void load_on_host(
    const specfem::point::simd_index<PartialDerivativesType::dimension> &index,
    const specfem::compute::partial_derivatives &derivatives,
    PartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int nspec = derivatives.nspec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PartialDerivativesType::store_jacobian;

  using simd = typename PartialDerivativesType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

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

template <typename PartialDerivativesType,
          typename std::enable_if_t<!PartialDerivativesType::simd::using_simd,
                                    int> = 0>
void load_on_host(
    const specfem::point::index<PartialDerivativesType::dimension> &index,
    const specfem::compute::partial_derivatives &derivatives,
    PartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PartialDerivativesType::store_jacobian;

  partial_derivatives.xix = derivatives.h_xix(ispec, iz, ix);
  partial_derivatives.gammax = derivatives.h_gammax(ispec, iz, ix);
  partial_derivatives.xiz = derivatives.h_xiz(ispec, iz, ix);
  partial_derivatives.gammaz = derivatives.h_gammaz(ispec, iz, ix);
  if constexpr (StoreJacobian) {
    partial_derivatives.jacobian = derivatives.h_jacobian(ispec, iz, ix);
  }
}

template <typename PartialDerivativesType,
          typename std::enable_if_t<PartialDerivativesType::simd::using_simd,
                                    int> = 0>
void store_on_host(
    const specfem::point::simd_index<PartialDerivativesType::dimension> &index,
    const specfem::compute::partial_derivatives &derivatives,
    const PartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int nspec = derivatives.nspec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PartialDerivativesType::store_jacobian;

  using simd = typename PartialDerivativesType::simd;
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

template <typename PartialDerivativesType,
          typename std::enable_if_t<!PartialDerivativesType::simd::using_simd,
                                    int> = 0>
void store_on_host(
    const specfem::point::index<PartialDerivativesType::dimension> &index,
    const specfem::compute::partial_derivatives &derivatives,
    const PartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PartialDerivativesType::store_jacobian;

  derivatives.h_xix(ispec, iz, ix) = partial_derivatives.xix;
  derivatives.h_gammax(ispec, iz, ix) = partial_derivatives.gammax;
  derivatives.h_xiz(ispec, iz, ix) = partial_derivatives.xiz;
  derivatives.h_gammaz(ispec, iz, ix) = partial_derivatives.gammaz;
  if constexpr (StoreJacobian) {
    derivatives.h_jacobian(ispec, iz, ix) = partial_derivatives.jacobian;
  }
}

// template <bool load_jacobian>
// KOKKOS_FUNCTION void load_on_device(
//     const specfem::point::index &index,
//     const specfem::compute::partial_derivatives &derivatives,
//     specfem::point::partial_derivatives2<load_jacobian> &partial_derivatives)
//     {

//   const int ispec = index.ispec;
//   const int nspec = derivatives.nspec;
//   const int iz = index.iz;
//   const int ix = index.ix;

//   using simd = specfem::point::partial_derivatives2<load_jacobian>::simd;

//   if constexpr (using_simd) {
//     simd::mask_type mask(
//         [](std::size_t lane) { return ispec + int(lane) < nspec; });

//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .xix.copy_from(&derivatives.xix(ispec, iz, ix), simd::tag_type());
//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .gammax.copy_from(&derivatives.gammax(ispec, iz, ix),
//         simd::tag_type());
//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .xiz.copy_from(&derivatives.xiz(ispec, iz, ix), simd::tag_type());
//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .gammaz.copy_from(&derivatives.gammaz(ispec, iz, ix),
//         simd::tag_type());
//     if constexpr (load_jacobian) {
//       Kokkos::Experimental::where(mask, partial_derivatives)
//           .jacobian.copy_from(derivatives.jacobian(ispec, iz, ix),
//                               simd::tag_type());
//     }
//   } else {
//     partial_derivatives.xix = derivatives.xix(ispec, iz, ix);
//     partial_derivatives.gammax = derivatives.gammax(ispec, iz, ix);
//     partial_derivatives.xiz = derivatives.xiz(ispec, iz, ix);
//     partial_derivatives.gammaz = derivatives.gammaz(ispec, iz, ix);
//     if constexpr (load_jacobian) {
//       partial_derivatives.jacobian = derivatives.jacobian(ispec, iz, ix);
//     }
//   }

//   return;
// }

// template <bool StoreJacobian>
// void load_on_host(
//     const specfem::point::index &index,
//     const specfem::compute::partial_derivatives &derivatives,
//     specfem::point::partial_derivatives2<StoreJacobian> &partial_derivatives)
//     {

//   const int ispec = index.ispec;
//   const int nspec = derivatives.nspec;
//   const int iz = index.iz;
//   const int ix = index.ix;

//   using simd = specfem::point::partial_derivatives2<load_jacobian>::simd;

//   if constexpr (using_simd) {
//     simd::mask_type mask(
//         [](std::size_t lane) { return ispec + int(lane) < nspec; });

//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .xix.copy_from(&derivatives.h_xix(ispec, iz, ix), simd::tag_type());
//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .gammax.copy_from(&derivatives.h_gammax(ispec, iz, ix),
//                           simd::tag_type());
//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .xiz.copy_from(&derivatives.h_xiz(ispec, iz, ix), simd::tag_type());
//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .gammaz.copy_from(&derivatives.h_gammaz(ispec, iz, ix),
//                           simd::tag_type());
//     if constexpr (load_jacobian) {
//       Kokkos::Experimental::where(mask, partial_derivatives)
//           .jacobian.copy_from(derivatives.h_jacobian(ispec, iz, ix),
//                               simd::tag_type());
//     }
//   } else {
//     partial_derivatives.xix = derivatives.h_xix(ispec, iz, ix);
//     partial_derivatives.gammax = derivatives.h_gammax(ispec, iz, ix);
//     partial_derivatives.xiz = derivatives.h_xiz(ispec, iz, ix);
//     partial_derivatives.gammaz = derivatives.h_gammaz(ispec, iz, ix);
//     if constexpr (load_jacobian) {
//       partial_derivatives.jacobian = derivatives.h_jacobian(ispec, iz, ix);
//     }
//   }

//   return;
// }

// template <bool StoreJacobian>
// void update_on_host(const specfem::point::index &index,
//                     const specfem::compute::partial_derivatives &derivatives,
//                     const specfem::point::partial_derivatives2<StoreJacobian>
//                         &partial_derivatives) {

//   const int ispec = index.ispec;
//   const int nspec = derivatives.nspec;
//   const int iz = index.iz;
//   const int ix = index.ix;

//   using simd = specfem::point::partial_derivatives2<load_jacobian>::simd;

//   if constexpr (using_simd) {
//     simd::mask_type mask(
//         [](std::size_t lane) { return ispec + int(lane) < nspec; });

//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .xix.copy_to(&derivatives.h_xix(ispec, iz, ix), simd::tag_type());
//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .gammax.copy_to(&derivatives.h_gammax(ispec, iz, ix),
//         simd::tag_type());
//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .xiz.copy_to(&derivatives.h_xiz(ispec, iz, ix), simd::tag_type());
//     Kokkos::Experimental::where(mask, partial_derivatives)
//         .gammaz.copy_to(&derivatives.h_gammaz(ispec, iz, ix),
//         simd::tag_type());
//     if constexpr (load_jacobian) {
//       Kokkos::Experimental::where(mask, partial_derivatives)
//           .jacobian.copy_to(&derivatives.h_jacobian(ispec, iz, ix),
//                             simd::tag_type());
//     }
//   } else {
//     partial_derivatives.xix = derivatives.h_xix(ispec, iz, ix);
//     partial_derivatives.gammax = derivatives.h_gammax(ispec, iz, ix);
//     partial_derivatives.xiz = derivatives.h_xiz(ispec, iz, ix);
//     partial_derivatives.gammaz = derivatives.h_gammaz(ispec, iz, ix);
//     if constexpr (load_jacobian) {
//       partial_derivatives.jacobian = derivatives.h_jacobian(ispec, iz, ix);
//     }
//   }

//   return;

//   return;
// }

} // namespace compute
} // namespace specfem

#endif
