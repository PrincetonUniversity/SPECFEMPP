#ifndef _ALGORITHMS_GRADIENT_HPP
#define _ALGORITHMS_GRADIENT_HPP

#include "kokkos_abstractions.h"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {
template <typename MemberType, typename ViewType, typename QuadratureType,
          typename CallbackFunctor,
          std::enable_if_t<ViewType::isChunkViewType, int> = 0,
          std::enable_if_t<
              Kokkos::SpaceAccessibility<
                  typename MemberType::execution_space::scratch_memory_space,
                  typename ViewType::memory_space>::accessible,
              int> = 0>
KOKKOS_FUNCTION void gradient(
    const MemberType &team,
    const Kokkos::View<
        int *, typename MemberType::execution_space::memory_space> &indices,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const QuadratureType &quadrature, const ViewType &f,
    CallbackFunctor callback) {
  constexpr int components = ViewType::components;

  const int number_elements = indices.extent(0);

  constexpr int NGLL = ViewType::ngll;

  using VectorPointViewType =
      specfem::datatype::VectorPointViewType<type_real, 2, components>;

  static_assert(
      std::is_invocable_v<CallbackFunctor, int, specfem::point::index,
                          VectorPointViewType>,
      "CallbackFunctor must be invocable with the following signature: "
      "void(const int, const specfem::point::index, const "
      "specfem::kokkos::array_type<type_real, components>, const "
      "specfem::kokkos::array_type<type_real, components>)");

  constexpr int ThreadTile = 1;
  constexpr int NTiles = NGLL / ThreadTile + (NGLL % ThreadTile != 0);

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NTiles * NTiles * number_elements),
      [=](const int &ixz) {
        const int ielement = ixz % number_elements;
        const int xz = ixz / number_elements;
        const int iz = xz / NTiles * ThreadTile;
        const int ix = xz % NTiles * ThreadTile;

        const int ispec = indices(ielement);

        type_real df_dxi[components * ThreadTile * ThreadTile] = { 0.0 };
        type_real df_dgamma[components * ThreadTile * ThreadTile] = { 0.0 };

        type_real l_f[ThreadTile] = { 0.0 };
        type_real l_quad[ThreadTile] = { 0.0 };

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          for (int l = 0; l < NGLL; ++l) {
            for (int Tid = 0; Tid < ThreadTile; ++Tid) {
              l_f[Tid] = (Tid + iz < NGLL)
                             ? f(ielement, iz + Tid, l, icomponent)
                             : 0.0;
              l_quad[Tid] = (Tid + ix < NGLL) ? quadrature(ix + Tid, l) : 0.0;
            }

            for (int Tiz = 0; Tiz < ThreadTile; ++Tiz) {
              for (int Tix = 0; Tix < ThreadTile; ++Tix) {
                df_dxi[icomponent * ThreadTile * ThreadTile + Tiz * ThreadTile +
                       Tix] += l_quad[Tix] * l_f[Tiz];
              }
            }

            for (int Tid = 0; Tid < ThreadTile; ++Tid) {
              l_f[Tid] = (Tid + ix < NGLL)
                             ? f(ielement, l, ix + Tid, icomponent)
                             : 0.0;
              l_quad[Tid] = (Tid + iz < NGLL) ? quadrature(iz + Tid, l) : 0.0;
            }

            for (int Tiz = 0; Tiz < ThreadTile; ++Tiz) {
              for (int Tix = 0; Tix < ThreadTile; ++Tix) {
                df_dgamma[icomponent * ThreadTile * ThreadTile +
                          Tiz * ThreadTile + Tix] += l_quad[Tiz] * l_f[Tix];
              }
            }
          }
        }

        for (int Tiz = 0; Tiz < ThreadTile; ++Tiz) {
          for (int Tix = 0; Tix < ThreadTile; ++Tix) {
            if (iz + Tiz < NGLL && ix + Tix < NGLL) {
              const specfem::point::index index(ispec, iz + Tiz, ix + Tix);
              const auto point_partial_derivatives = [&]() {
                specfem::point::partial_derivatives2<false> result;
                specfem::compute::load_on_device(index, partial_derivatives,
                                                 result);
                return result;
              }();
              VectorPointViewType df;
              for (int icomponent = 0; icomponent < components; ++icomponent) {
                df(0, icomponent) =
                    df_dxi[icomponent * ThreadTile * ThreadTile +
                           Tiz * ThreadTile + Tix] *
                        point_partial_derivatives.xix +
                    df_dgamma[icomponent * ThreadTile * ThreadTile +
                              Tiz * ThreadTile + Tix] *
                        point_partial_derivatives.gammax;

                df(1, icomponent) =
                    df_dxi[icomponent * ThreadTile * ThreadTile +
                           Tiz * ThreadTile + Tix] *
                        point_partial_derivatives.xiz +
                    df_dgamma[icomponent * ThreadTile * ThreadTile +
                              Tiz * ThreadTile + Tix] *
                        point_partial_derivatives.gammaz;
              }
              callback(ielement, index, df);
            }
          }
        }

        // for (int l = 0; l < NGLL; ++l) {
        //   for (int icomponent = 0; icomponent < components; ++icomponent) {
        //     df_dxi[icomponent] +=
        //         quadrature(ix, l) * f(ielement, iz, l, icomponent);
        //     df_dgamma[icomponent] +=
        //         quadrature(iz, l) * f(ielement, l, ix, icomponent);
        //   }
        // }

        // const auto point_partial_derivatives = [&]() {
        //   specfem::point::partial_derivatives2<false> result;
        //   specfem::compute::load_on_device(index, partial_derivatives,
        //   result); return result;
        // }();

        // VectorPointViewType df;

        // for (int icomponent = 0; icomponent < components; ++icomponent) {
        //   df(0, icomponent) =
        //       point_partial_derivatives.xix * df_dxi[icomponent] +
        //       point_partial_derivatives.gammax * df_dgamma[icomponent];

        //   df(1, icomponent) =
        //       point_partial_derivatives.xiz * df_dxi[icomponent] +
        //       point_partial_derivatives.gammaz * df_dgamma[icomponent];
        // }

        // callback(ielement, index, df);
      });

  return;
}

template <typename MemberType, typename ViewType, typename QuadratureType,
          typename CallbackFunctor,
          std::enable_if_t<ViewType::isChunkViewType, int> = 0,
          std::enable_if_t<Kokkos::SpaceAccessibility<
                               typename MemberType::execution_space,
                               typename ViewType::memory_space>::accessible,
                           int> = 0>
KOKKOS_FUNCTION void gradient(
    const MemberType &team,
    const Kokkos::View<
        int *, typename MemberType::execution_space::memory_space> &indices,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const QuadratureType &quadrature, const ViewType &f, const ViewType &g,
    CallbackFunctor callback) {
  constexpr int components = ViewType::components;

  const int number_elements = indices.extent(0);

  constexpr int NGLL = ViewType::ngll;

  using VectorPointViewType =
      specfem::datatype::VectorPointViewType<type_real, 2, components>;

  static_assert(
      std::is_invocable_v<CallbackFunctor, int, specfem::point::index,
                          VectorPointViewType, VectorPointViewType>,
      "CallbackFunctor must be invocable with the following signature: "
      "void(const int, const specfem::point::index, const "
      "pecfem::datatype::VectorPointViewType<type_real, 2, components>, "
      "const "
      "pecfem::datatype::VectorPointViewType<type_real, 2, components>)");

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL * number_elements),
      [=](const int &ixz) {
        const int ielement = ixz % number_elements;
        const int xz = ixz / number_elements;
        int ix, iz;
        sub2ind(xz, NGLL, iz, ix);

        const int ispec = indices(ielement);

        specfem::point::index index(ispec, iz, ix);

        type_real df_dxi[components];
        type_real df_dgamma[components];

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          df_dxi[icomponent] = 0.0;
          df_dgamma[icomponent] = 0.0;
        }

        for (int l = 0; l < NGLL; ++l) {
          for (int icomponent = 0; icomponent < components; ++icomponent) {
            df_dxi[icomponent] +=
                quadrature(ix, l) * f(ielement, iz, l, icomponent);
            df_dgamma[icomponent] +=
                quadrature(iz, l) * f(ielement, l, ix, icomponent);
          }
        }

        const auto point_partial_derivatives = [&]() {
          specfem::point::partial_derivatives2<false> result;
          specfem::compute::load_on_device(index, partial_derivatives, result);
          return result;
        }();

        VectorPointViewType df;

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          df(0, icomponent) =
              point_partial_derivatives.xix * df_dxi[icomponent] +
              point_partial_derivatives.gammax * df_dgamma[icomponent];

          df(1, icomponent) =
              point_partial_derivatives.xiz * df_dxi[icomponent] +
              point_partial_derivatives.gammaz * df_dgamma[icomponent];
        }

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          df_dxi[icomponent] = 0.0;
          df_dgamma[icomponent] = 0.0;
        }

        for (int l = 0; l < NGLL; ++l) {
          for (int icomponent = 0; icomponent < components; ++icomponent) {
            df_dxi[icomponent] +=
                quadrature(ix, l) * g(ielement, iz, l, icomponent);
            df_dgamma[icomponent] +=
                quadrature(iz, l) * g(ielement, l, ix, icomponent);
          }
        }

        VectorPointViewType dg;

        for (int icomponent = 0; icomponent < components; ++icomponent) {
          dg(0, icomponent) =
              point_partial_derivatives.xix * df_dxi[icomponent] +
              point_partial_derivatives.gammax * df_dgamma[icomponent];

          dg(1, icomponent) =
              point_partial_derivatives.xiz * df_dxi[icomponent] +
              point_partial_derivatives.gammaz * df_dgamma[icomponent];
        }

        callback(ielement, index, df, dg);
      });

  return;
}

// template <int NGLL, int components, typename Layout, typename MemorySpace,
//           typename MemoryTraits>
// KOKKOS_FUNCTION void
// gradient(const int ix, const int iz,
//          const Kokkos::View<type_real[NGLL][NGLL], Layout, MemorySpace,
//                             MemoryTraits> &hprime,
//          const Kokkos::View<type_real[NGLL][NGLL][components], Layout,
//                             MemorySpace, MemoryTraits> &function,
//          const specfem::point::partial_derivatives2<false>
//          partial_derivatives, specfem::kokkos::array_type<type_real,
//          components> &dfield_dx, specfem::kokkos::array_type<type_real,
//          components> &dfield_dz) {
//   specfem::kokkos::array_type<type_real, components> dfield_dxi;
//   specfem::kokkos::array_type<type_real, components> dfield_dgamma;

// #ifdef KOKKOS_ENABLE_CUDA
// #pragma unroll
// #endif
//   for (int l = 0; l < NGLL; ++l) {
// #ifdef KOKKOS_ENABLE_CUDA
// #pragma unroll
// #endif
//     for (int icomponent = 0; icomponent < components; ++icomponent) {
//       dfield_dxi[icomponent] += hprime(ix, l) * function(iz, l,
//       icomponent); dfield_dgamma[icomponent] += hprime(iz, l) * function(l,
//       ix, icomponent);
//     }
//   }

// #ifdef KOKKOS_ENABLE_CUDA
// #pragma unroll
// #endif
//   for (int icomponent = 0; icomponent < components; ++icomponent) {
//     dfield_dx[icomponent] =
//         partial_derivatives.xix * dfield_dxi[icomponent] +
//         partial_derivatives.gammax * dfield_dgamma[icomponent];
//     dfield_dz[icomponent] =
//         partial_derivatives.xiz * dfield_dxi[icomponent] +
//         partial_derivatives.gammaz * dfield_dgamma[icomponent];
//   }

//   return;
// }

// template <int NGLL, int components, typename Layout, typename MemorySpace,
//           typename MemoryTraits, typename CallbackFunctor>
// KOKKOS_FUNCTION void
// gradient(const int ix, const int iz,
//          const Kokkos::View<type_real[NGLL][NGLL], Layout, MemorySpace,
//                             MemoryTraits> &hprime,
//          const Kokkos::View<type_real[NGLL][NGLL][components], Layout,
//                             MemorySpace, MemoryTraits> &function,
//          const specfem::point::partial_derivatives2<false>
//          partial_derivatives, specfem::kokkos::array_type<type_real,
//          components> &dfield_dx, specfem::kokkos::array_type<type_real,
//          components> &dfield_dz, CallbackFunctor callback) {
//   gradient(ix, iz, hprime, function, partial_derivatives, dfield_dx,
//   dfield_dz); callback(dfield_dx, dfield_dz);

//   return;
// }

// template <int NGLL, int components, typename ExecutionSpace, typename
// Layout,
//           typename MemorySpace,
//           std::enable_if_t<Kokkos::SpaceAccessibility<ExecutionSpace,
//                                                       MemorySpace>::accessible,
//                            int> = 0>
// KOKKOS_FUNCTION void
// gradient(const typename Kokkos::TeamPolicy<ExecutionSpace>::memory_space
// &team,
//          const int ispec,
//          const Kokkos::View<type_real[NGLL][NGLL], Layout, MemorySpace>
//          &hprime, const Kokkos::View<type_real[NGLL][NGLL][components],
//          Layout,
//                             MemorySpace> &function,
//          const specfem::compute::partial_derivatives &partial_derivatives,
//          Kokkos::View<type_real[NGLL][NGLL][components], Layout,
//          MemorySpace>
//              &dfield_dx,
//          Kokkos::View<type_real[NGLL][NGLL][components], Layout,
//          MemorySpace>
//              &dfield_dz) {
//   Kokkos::parallel_for(
//       "specfem::algorithms::gradient",
//       Kokkos::TeamThreadRange(team, NGLL * NGLL), [=](const int &xz) {
//         int ix, iz;
//         sub2ind(xz, NGLL, iz, ix);

//         specfem::kokkos::array_type<type_real, components> dfield_dxi;
//         specfem::kokkos::array_type<type_real, components> dfield_dgamma;

// #ifdef KOKKOS_ENABLE_CUDA
// #pragma unroll
// #endif
//         for (int l = 0; l < NGLL; ++l) {
// #ifdef KOKKOS_ENABLE_CUDA
// #pragma unroll
// #endif
//           for (int icomponent = 0; icomponent < components; ++icomponent) {
//             dfield_dxi[icomponent] +=
//                 hprime(ix, l) * function(iz, l, icomponent);
//             dfield_dgamma[icomponent] +=
//                 hprime(iz, l) * function(l, ix, icomponent);
//           }
//         }

//         specfem::point::partial_derivatives2<false>
//         point_partial_derivatives; const specfem::point::index index(ispec,
//         iz, ix); specfem::compute::load_on_device(index,
//         partial_derivatives,
//                                          point_partial_derivatives);

// #ifdef KOKKOS_ENABLE_CUDA
// #pragma unroll
// #endif
//         for (int icomponent = 0; icomponent < components; ++icomponent) {
//           dfield_dx(iz, ix, icomponent) =
//               point_partial_derivatives.xix * dfield_dxi[icomponent] +
//               point_partial_derivatives.gammax * dfield_dgamma[icomponent];
//           dfield_dz(iz, ix, icomponent) =
//               point_partial_derivatives.xiz * dfield_dxi[icomponent] +
//               point_partial_derivatives.gammaz * dfield_dgamma[icomponent];
//         }
//       });

//   return;
// }

// template <int NGLL, int components, typename ExecutionSpace, typename
// Layout,
//           typename MemorySpace, typename CallbackFunctor,
//           std::enable_if_t<Kokkos::SpaceAccessibility<ExecutionSpace,
//                                                       MemorySpace>::accessible,
//                            int> = 0>
// KOKKOS_FUNCTION void
// gradient(const typename Kokkos::TeamPolicy<ExecutionSpace>::memory_space
// &team,
//          const int ispec,
//          const Kokkos::View<type_real[NGLL][NGLL], Layout, MemorySpace>
//          &hprime, const Kokkos::View<type_real[NGLL][NGLL][components],
//          Layout,
//                             MemorySpace> &function,
//          const specfem::compute::partial_derivatives &partial_derivatives,
//          Kokkos::View<type_real[NGLL][NGLL][components], Layout,
//          MemorySpace>
//              &dfield_dx,
//          Kokkos::View<type_real[NGLL][NGLL][components], Layout,
//          MemorySpace>
//              &dfield_dz,
//          CallbackFunctor callback) {
//   gradient(team, ispec, hprime, function, partial_derivatives, dfield_dx,
//            dfield_dz);

//   Kokkos::parallel_for("specfem::algorithms::gradient::callback",
//                        Kokkos::TeamThreadRange(team, NGLL * NGLL),
//                        [=](const int &xz) {
//                          int ix, iz;
//                          sub2ind(xz, NGLL, iz, ix);
//                          callback(iz, ix);
//                        });

//   return;
// }

} // namespace algorithms
} // namespace specfem

#endif /* _ALGORITHMS_GRADIENT_HPP */
