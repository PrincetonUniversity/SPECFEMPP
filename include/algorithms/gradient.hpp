#ifndef _ALGORITHMS_GRADIENT_HPP
#define _ALGORITHMS_GRADIENT_HPP

#include "kokkos_abstractions.h"
#include "point/partial_derivatives.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

template <int NGLL, int components, typename Layout, typename MemorySpace,
          typename MemoryTraits>
KOKKOS_FUNCTION void
gradient(const int ix, const int iz,
         const Kokkos::View<type_real[NGLL][NGLL], Layout, MemorySpace,
                            MemoryTraits> &hprime,
         const Kokkos::View<type_real[NGLL][NGLL][components], Layout,
                            MemorySpace, MemoryTraits> &function,
         const specfem::point::partial_derivatives2<false> partial_derivatives,
         specfem::kokkos::array_type<type_real, components> &dfield_dx,
         specfem::kokkos::array_type<type_real, components> &dfield_dz) {

  specfem::kokkos::array_type<type_real, components> dfield_dxi;
  specfem::kokkos::array_type<type_real, components> dfield_dgamma;

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; ++l) {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      dfield_dxi[icomponent] += hprime(ix, l) * function(iz, l, icomponent);
      dfield_dgamma[icomponent] += hprime(iz, l) * function(l, ix, icomponent);
    }
  }

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int icomponent = 0; icomponent < components; ++icomponent) {
    dfield_dx[icomponent] =
        partial_derivatives.xix * dfield_dxi[icomponent] +
        partial_derivatives.gammax * dfield_dgamma[icomponent];
    dfield_dz[icomponent] =
        partial_derivatives.xiz * dfield_dxi[icomponent] +
        partial_derivatives.gammaz * dfield_dgamma[icomponent];
  }

  return;
}

template <int NGLL, int components, typename Layout, typename MemorySpace,
          typename MemoryTraits, typename CallbackFunctor>
KOKKOS_FUNCTION void
gradient(const int ix, const int iz,
         const Kokkos::View<type_real[NGLL][NGLL], Layout, MemorySpace,
                            MemoryTraits> &hprime,
         const Kokkos::View<type_real[NGLL][NGLL][components], Layout,
                            MemorySpace, MemoryTraits> &function,
         const specfem::point::partial_derivatives2<false> partial_derivatives,
         specfem::kokkos::array_type<type_real, components> &dfield_dx,
         specfem::kokkos::array_type<type_real, components> &dfield_dz,
         CallbackFunctor callback) {

  gradient(ix, iz, hprime, function, partial_derivatives, dfield_dx, dfield_dz);
  callback(dfield_dx, dfield_dz);

  return;
}

template <int NGLL, int components, typename ExecutionSpace, typename Layout,
          typename MemorySpace,
          std::enable_if_t<Kokkos::SpaceAccessibility<ExecutionSpace,
                                                      MemorySpace>::accessible,
                           int> = 0>
KOKKOS_FUNCTION void
gradient(const typename Kokkos::TeamPolicy<ExecutionSpace>::memory_space &team,
         const int ispec,
         const Kokkos::View<type_real[NGLL][NGLL], Layout, MemorySpace> &hprime,
         const Kokkos::View<type_real[NGLL][NGLL][components], Layout,
                            MemorySpace> &function,
         const specfem::compute::partial_derivatives &partial_derivatives,
         Kokkos::View<type_real[NGLL][NGLL][components], Layout, MemorySpace>
             &dfield_dx,
         Kokkos::View<type_real[NGLL][NGLL][components], Layout, MemorySpace>
             &dfield_dz) {

  Kokkos::parallel_for(
      "specfem::algorithms::gradient",
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [=](const int &xz) {
        int ix, iz;
        sub2ind(xz, NGLL, iz, ix);

        specfem::kokkos::array_type<type_real, components> dfield_dxi;
        specfem::kokkos::array_type<type_real, components> dfield_dgamma;

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
        for (int l = 0; l < NGLL; ++l) {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
          for (int icomponent = 0; icomponent < components; ++icomponent) {
            dfield_dxi[icomponent] +=
                hprime(ix, l) * function(iz, l, icomponent);
            dfield_dgamma[icomponent] +=
                hprime(iz, l) * function(l, ix, icomponent);
          }
        }

        specfem::point::partial_derivatives2<false> point_partial_derivatives;
        const specfem::point::index index(ispec, iz, ix);
        specfem::compute::load_on_device(index, partial_derivatives,
                                         point_partial_derivatives);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
        for (int icomponent = 0; icomponent < components; ++icomponent) {
          dfield_dx(iz, ix, icomponent) =
              point_partial_derivatives.xix * dfield_dxi[icomponent] +
              point_partial_derivatives.gammax * dfield_dgamma[icomponent];
          dfield_dz(iz, ix, icomponent) =
              point_partial_derivatives.xiz * dfield_dxi[icomponent] +
              point_partial_derivatives.gammaz * dfield_dgamma[icomponent];
        }
      });

  return;
}

template <int NGLL, int components, typename ExecutionSpace, typename Layout,
          typename MemorySpace, typename CallbackFunctor,
          std::enable_if_t<Kokkos::SpaceAccessibility<ExecutionSpace,
                                                      MemorySpace>::accessible,
                           int> = 0>
KOKKOS_FUNCTION void
gradient(const typename Kokkos::TeamPolicy<ExecutionSpace>::memory_space &team,
         const int ispec,
         const Kokkos::View<type_real[NGLL][NGLL], Layout, MemorySpace> &hprime,
         const Kokkos::View<type_real[NGLL][NGLL][components], Layout,
                            MemorySpace> &function,
         const specfem::compute::partial_derivatives &partial_derivatives,
         Kokkos::View<type_real[NGLL][NGLL][components], Layout, MemorySpace>
             &dfield_dx,
         Kokkos::View<type_real[NGLL][NGLL][components], Layout, MemorySpace>
             &dfield_dz,
         CallbackFunctor callback) {

  gradient(team, ispec, hprime, function, partial_derivatives, dfield_dx,
           dfield_dz);

  Kokkos::parallel_for("specfem::algorithms::gradient::callback",
                       Kokkos::TeamThreadRange(team, NGLL * NGLL),
                       [=](const int &xz) {
                         int ix, iz;
                         sub2ind(xz, NGLL, iz, ix);
                         callback(iz, ix);
                       });

  return;
}

} // namespace algorithms
} // namespace specfem

#endif /* _ALGORITHMS_GRADIENT_HPP */
