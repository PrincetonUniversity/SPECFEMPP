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
void divergence(
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

  // Compute the integral
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL * number_elements),
      [=](const int &ixz) {
        const int ielement = ixz % number_elements;
        const int xz = ixz / number_elements;
        int ix, iz;
        sub2ind(xz, NGLL, iz, ix);

        const int ispec = indices(ielement);

        specfem::point::index index(ispec, iz, ix);

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

        specfem::datatype::ScalarPointViewType<type_real, components> result;

        for (int icomp = 0; icomp < components; ++icomp) {
          result(icomp) =
              weights[ix] * temp1l[icomp] + weights[iz] * temp2l[icomp];
        }

        callback(ielement, index, result);
      });

  return;
}

} // namespace algorithms
} // namespace specfem
