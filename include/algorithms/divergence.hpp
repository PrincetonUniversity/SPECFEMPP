#pragma once

#include "compute/compute_partial_derivatives.hpp"
#include "datatypes/point_view.hpp"
#include "point/coordinates.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

template <
    typename MemberType, typename VectorFieldType, typename QuadratureType,
    typename CallableType,
    std::enable_if_t<(VectorFieldType::isChunkViewType &&
                      VectorFieldType::isVectorViewType),
                     int> = 0,
    std::enable_if_t<Kokkos::SpaceAccessibility<
                         typename MemberType::execution_space,
                         typename VectorFieldType::memory_space>::accessible,
                     int> = 0>
KOKKOS_FUNCTION void divergence(
    const MemberType &team,
    const Kokkos::View<
        int *, typename MemberType::execution_space::memory_space> &indices,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const Kokkos::View<type_real *,
                       typename MemberType::execution_space::memory_space>
        &weights,
    const QuadratureType &hprimewgll, const VectorFieldType &f,
    CallableType callback) {

  constexpr int components = VectorFieldType::components;
  constexpr int NGLL = VectorFieldType::ngll;
  const int number_elements = indices.extent(0);

  constexpr bool is_host_space =
      std::is_same<typename MemberType::execution_space::memory_space,
                   Kokkos::HostSpace>::value;

  using ScalarPointViewType =
      specfem::datatype::ScalarPointViewType<type_real, components>;

  static_assert(
      std::is_invocable_v<CallableType, int, specfem::point::index,
                          ScalarPointViewType>,
      "CallableType must be invocable with arguments (int, "
      "specfem::point::index, "
      "specfem::datatype::ScalarPointViewType<type_real, components>)");

  // Compute the integral
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL * number_elements),
      [=](const int &ixz) {
        const int ielement = ixz % number_elements;
        const int xz = ixz / number_elements;
        const int iz = xz / NGLL;
        const int ix = xz % NGLL;

        const int ispec = indices(ielement);

        // type_real temp1l[components] = { 0.0 };
        // type_real temp2l[components] = { 0.0 };

        specfem::point::index index(ispec, iz, ix);

        // type_real l_f[ThreadTile] = { 0.0 };
        // type_real l_quad[ThreadTile] = { 0.0 };

        // for (int icomponent = 0; icomponent < components; ++icomponent) {
        //   for (int l = 0; l < NGLL; ++l) {
        //     for (int Tid = 0; Tid < ThreadTile; ++Tid) {
        //       l_f[Tid] = (Tid + iz < NGLL)
        //                      ? f(ielement, iz + Tid, l, 0, icomponent)
        //                      : 0.0;
        //       l_quad[Tid] = (Tid + ix < NGLL) ? hprimewgll(ix + Tid, l) :
        //       0.0;
        //     }

        //     for (int Tiz = 0; Tiz < ThreadTile; ++Tiz) {
        //       for (int Tix = 0; Tix < ThreadTile; ++Tix) {
        //         temp1l[icomponent * ThreadTile * ThreadTile + Tiz *
        //         ThreadTile +
        //                Tix] += l_f[Tiz] * l_quad[Tix];
        //       }
        //     }

        //     for (int Tid = 0; Tid < ThreadTile; ++Tid) {
        //       l_f[Tid] = (Tid + ix < NGLL)
        //                      ? f(ielement, l, ix + Tid, 1, icomponent)
        //                      : 0.0;
        //       l_quad[Tid] = (Tid + iz < NGLL) ? hprimewgll(iz + Tid, l) :
        //       0.0;
        //     }

        //     for (int Tiz = 0; Tiz < ThreadTile; ++Tiz) {
        //       for (int Tix = 0; Tix < ThreadTile; ++Tix) {
        //         temp2l[icomponent * ThreadTile * ThreadTile + Tix *
        //         ThreadTile +
        //                Tiz] += l_f[Tix] * l_quad[Tiz];
        //       }
        //     }
        //   }
        // }

        // for (int Tiz = 0; Tiz < ThreadTile; ++Tiz) {
        //   for (int Tix = 0; Tix < ThreadTile; ++Tix) {
        //     if (iz + Tiz < NGLL && ix + Tix < NGLL) {
        //       const specfem::point::index index(ispec, iz + Tiz, ix + Tix);
        //       const type_real jacobian = [&]() {
        //         if constexpr (is_host_space) {
        //           return partial_derivatives.h_jacobian(ispec, iz + Tiz,
        //                                                 ix + Tix);
        //         } else {
        //           return partial_derivatives.jacobian(ispec, iz + Tiz,
        //                                               ix + Tix);
        //         }
        //       }();

        //       ScalarPointViewType result;
        //       for (int icomp = 0; icomp < components; ++icomp) {
        //         result(icomp) = (weights(iz + Tiz) *
        //                              temp1l[icomp * ThreadTile * ThreadTile +
        //                                     Tiz * ThreadTile + Tix] +
        //                          weights(ix + Tix) *
        //                              temp2l[icomp * ThreadTile * ThreadTile +
        //                                     Tiz * ThreadTile + Tix]) *
        //                         jacobian;
        //       }

        //       callback(ielement, index, result);
        //     }
        //   }
        // }

        const type_real jacobian = [&]() {
          if constexpr (is_host_space) {
            return partial_derivatives.h_jacobian(ispec, iz, ix);
          } else {
            return partial_derivatives.jacobian(ispec, iz, ix);
          }
        }();

        type_real temp1l[components] = { 0.0 };
        type_real temp2l[components] = { 0.0 };

        for (int l = 0; l < NGLL; ++l) {
          for (int icomp = 0; icomp < components; ++icomp) {
            temp1l[icomp] +=
                f(ielement, iz, l, 0, icomp) * hprimewgll(ix, l) * jacobian;
            temp2l[icomp] +=
                f(ielement, l, ix, 1, icomp) * hprimewgll(iz, l) * jacobian;
          }
        }

        ScalarPointViewType result;

        for (int icomp = 0; icomp < components; ++icomp) {
          result(icomp) =
              weights(iz) * temp1l[icomp] + weights(ix) * temp2l[icomp];
        }

        callback(ielement, index, result);
      });

  return;
}

} // namespace algorithms
} // namespace specfem
