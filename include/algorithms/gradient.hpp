#ifndef _ALGORITHMS_GRADIENT_HPP
#define _ALGORITHMS_GRADIENT_HPP

#include "kokkos_abstractions.h"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

template <typename MemberType, typename FieldType, typename QuadratureType,
          typename CallbackFunctor,
          std::enable_if_t<
              Kokkos::SpaceAccessibility<
                  typename MemberType::execution_space::scratch_memory_space,
                  typename FieldType::memory_space>::accessible,
              int> = 0>
KOKKOS_FUNCTION void gradient(
    const MemberType &team,
    const Kokkos::View<
        int *, typename MemberType::execution_space::memory_space> &indices,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const QuadratureType &quadrature, const FieldType &f, const FieldType &g,
    CallbackFunctor callback) {

  constexpr auto MediumTag = FieldType::medium_tag;
  constexpr auto DimensionType = FieldType::dimension_type;

  using PointFieldDerivativesType =
      specfem::point::field_derivatives<DimensionType, MediumTag>;

  const int number_elements = indices.extent(0);

  constexpr int NGLL = QuadratureType::ngll;

  static_assert(QuadratureType::ngll == FieldType::ngll,
                "The number of GLL points in the quadrature and field must "
                "be the same");

  static_assert(
      std::is_invocable_v<CallbackFunctor, specfem::point::index,
                          PointFieldDerivativesType, PointFieldDerivativesType>,
      "CallbackFunctor must be invocable with the following signature: "
      "void(const specfem::point::index, const "
      "specfem::point::field_derivatives<DimensionType, MediumTag>, const "
      "specfem::point::field_derivatives<DimensionType, MediumTag>)");

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, number_elements), [=](const int &ielement) {
        const int ispec = indices(ielement);

        for (int ix = 0; ix < NGLL; ++ix) {
          for (int iz = 0; iz < NGLL; ++iz) {
            specfem::point::index index(ispec, iz, ix);

            type_real df_dxi[FieldType::components];
            type_real df_dgamma[FieldType::components];

            type_real dg_dxi[FieldType::components];
            type_real dg_dgamma[FieldType::components];

            for (int icomponent = 0; icomponent < FieldType::components;
                 ++icomponent) {
              df_dxi[icomponent] = 0.0;
              df_dgamma[icomponent] = 0.0;

              dg_dxi[icomponent] = 0.0;
              dg_dgamma[icomponent] = 0.0;
            }

            for (int l = 0; l < NGLL; ++l) {
              for (int icomponent = 0; icomponent < FieldType::components;
                   ++icomponent) {
                df_dxi[icomponent] +=
                    quadrature.hprime_gll(ix, l) *
                    f.displacement(ielement, icomponent, iz, l);
                df_dgamma[icomponent] +=
                    quadrature.hprime_gll(iz, l) *
                    f.displacement(ielement, icomponent, l, ix);

                dg_dxi[icomponent] +=
                    quadrature.hprime_gll(ix, l) *
                    g.displacement(ielement, icomponent, iz, l);
                dg_dgamma[icomponent] +=
                    quadrature.hprime_gll(iz, l) *
                    g.displacement(ielement, icomponent, l, ix);
              }
            }

            const auto point_partial_derivatives = [&]() {
              specfem::point::partial_derivatives2<false> result;
              specfem::compute::load_on_device(index, partial_derivatives,
                                               result);
              return result;
            }();

            PointFieldDerivativesType df;
            PointFieldDerivativesType dg;

            for (int icomponent = 0; icomponent < FieldType::components;
                 ++icomponent) {
              df.du_dx[icomponent] =
                  point_partial_derivatives.xix * df_dxi[icomponent] +
                  point_partial_derivatives.gammax * df_dgamma[icomponent];

              df.du_dz[icomponent] =
                  point_partial_derivatives.xiz * df_dxi[icomponent] +
                  point_partial_derivatives.gammaz * df_dgamma[icomponent];

              dg.du_dx[icomponent] =
                  point_partial_derivatives.xix * dg_dxi[icomponent] +
                  point_partial_derivatives.gammax * dg_dgamma[icomponent];

              dg.du_dz[icomponent] =
                  point_partial_derivatives.xiz * dg_dxi[icomponent] +
                  point_partial_derivatives.gammaz * dg_dgamma[icomponent];
            }

            callback(index, df, dg);
          }
        }
      });

  return;
}

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
