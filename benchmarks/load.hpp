#pragma once

#include "element/quadrature.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "point/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace benchmarks {

template <typename MemberType, typename ViewType>
KOKKOS_FUNCTION void
load_on_device(const MemberType &team,
               const specfem::compute::quadrature &quadrature,
               ViewType &element_quadrature) {

  constexpr bool store_hprime_gll = ViewType::store_hprime_gll;

  constexpr bool store_weight_times_hprime_gll =
      ViewType::store_weight_times_hprime_gll;
  constexpr int NGLL = ViewType::ngll;

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a host execution space");

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
        int ix, iz;
        sub2ind(xz, NGLL, iz, ix);
        if constexpr (store_hprime_gll) {
          element_quadrature.hprime_gll(iz, ix) = quadrature.gll.hprime(iz, ix);
        }
        if constexpr (store_weight_times_hprime_gll) {
          element_quadrature.hprime_wgll(ix, iz) =
              quadrature.gll.hprime(iz, ix) * quadrature.gll.weights(iz);
        }
      });
}

template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              PointPartialDerivativesType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
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

template <
    typename MemberType, typename WavefieldType, typename IteratorType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const MemberType &team, const IteratorType &iterator,
               const WavefieldType &field, ViewType &chunk_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;

  static_assert(
      std::is_same_v<typename IteratorType::simd, typename ViewType::simd>,
      "Iterator and View must have the same simd type");

  static_assert(
      Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                 typename ViewType::memory_space>::accessible,
      "Calling team must have access to the memory space of the view");

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a device execution space");

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int &i) {
        const auto iterator_index = iterator(i);
        const int ielement = iterator_index.ielement;
        const int ispec = iterator_index.index.ispec;
        const int iz = iterator_index.index.iz;
        const int ix = iterator_index.index.ix;

        for (int lane = 0; lane < IteratorType::simd::size(); ++lane) {
          if (!iterator_index.index.mask(lane)) {
            continue;
          }

          const int iglob = field.assembly_index_mapping(
              field.index_mapping(ispec + lane, iz, ix),
              static_cast<int>(MediumType));

          for (int icomp = 0; icomp < components; ++icomp) {
            if constexpr (StoreDisplacement) {
              chunk_field.displacement(ielement, iz, ix, icomp)[lane] =
                  curr_field.field(iglob, icomp);
            }
            if constexpr (StoreVelocity) {
              chunk_field.velocity(ielement, iz, ix, icomp)[lane] =
                  curr_field.field_dot(iglob, icomp);
            }
            if constexpr (StoreAcceleration) {
              chunk_field.acceleration(ielement, iz, ix, icomp)[lane] =
                  curr_field.field_dot_dot(iglob, icomp);
            }
            if constexpr (StoreMassMatrix) {
              chunk_field.mass_matrix(ielement, iz, ix, icomp)[lane] =
                  curr_field.mass_inverse(iglob, icomp);
            }
          }
        }
      });

  return;
}

} // namespace benchmarks
} // namespace specfem
