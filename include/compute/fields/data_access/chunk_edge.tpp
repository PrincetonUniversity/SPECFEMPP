#pragma once

#include "compute/fields/data_access.tpp"


namespace specfem {
namespace compute {

template <
    typename MemberType, typename WavefieldType, typename IteratorType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkEdgeFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load_on_device(const MemberType &team, const IteratorType &iterator,
                    const WavefieldType &field, ViewType &chunk_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;
  constexpr static bool using_simd = ViewType::simd::using_simd;

  static_assert(
      std::is_same_v<typename IteratorType::simd, typename ViewType::simd>,
      "Iterator and View must have the same simd type");

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a device execution space");

  static_assert(
      Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                 typename ViewType::memory_space>::accessible,
      "Calling team must have access to the memory space of the view");

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
      const int igll = iterator_index.igll;

      if constexpr(using_simd){
        for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
          if (!iterator_index.index.mask(lane)) {
            continue;
          }

          const int iglob = field.assembly_index_mapping(
              field.index_mapping(ispec + lane, iz, ix),
              static_cast<int>(MediumType));

          for (int icomp = 0; icomp < components; ++icomp) {
            if constexpr (StoreDisplacement) {
              chunk_field.displacement(ielement, igll, icomp)[lane] =
                  curr_field.field(iglob, icomp);
            }
            if constexpr (StoreVelocity) {
              chunk_field.velocity(ielement, igll, icomp)[lane] =
                  curr_field.field_dot(iglob, icomp);
            }
            if constexpr (StoreAcceleration) {
              chunk_field.acceleration(ielement, igll, icomp)[lane] =
                  curr_field.field_dot_dot(iglob, icomp);
            }
            if constexpr (StoreMassMatrix) {
              chunk_field.mass_matrix(ielement, igll, icomp)[lane] =
                  curr_field.mass_inverse(iglob, icomp);
            }
          }
        }
      }else{
        const int iglob = field.assembly_index_mapping(
          field.index_mapping(ispec, iz, ix), static_cast<int>(MediumType));

        for (int icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            chunk_field.displacement(ielement, igll, icomp) =
                curr_field.field(iglob, icomp);
          }
          if constexpr (StoreVelocity) {
            chunk_field.velocity(ielement, igll, icomp) =
                curr_field.field_dot(iglob, icomp);
          }
          if constexpr (StoreAcceleration) {
            chunk_field.acceleration(ielement, igll, icomp) =
                curr_field.field_dot_dot(iglob, icomp);
          }
          if constexpr (StoreMassMatrix) {
            chunk_field.mass_matrix(ielement, igll, icomp) =
                curr_field.mass_inverse(iglob, icomp);
          }
        }
      }
    });

  return;
}


template <
    typename MemberType, typename WavefieldType, typename IteratorType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkEdgeFieldType, int> = 0>
inline void impl_load_on_host(const MemberType &team, const IteratorType &iterator,
                       const WavefieldType &field, ViewType &chunk_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;
  constexpr static bool using_simd = ViewType::simd::using_simd;

  static_assert(
      std::is_same_v<typename IteratorType::simd, typename ViewType::simd>,
      "Iterator and View must have the same simd type");

  static_assert(
      std::is_same_v<typename MemberType::execution_space, Kokkos::HostSpace>,
      "Calling team must have a host execution space");

  static_assert(
      Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                 typename ViewType::memory_space>::accessible,
      "Calling team must have access to the memory space of the view");

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
      const int igll = iterator_index.igll;

      if constexpr(using_simd){
        for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
          if (!iterator_index.index.mask(lane)) {
            continue;
          }

          const int iglob = field.h_assembly_index_mapping(
              field.h_index_mapping(ispec + lane, iz, ix),
              static_cast<int>(MediumType));

          for (int icomp = 0; icomp < components; ++icomp) {
            if constexpr (StoreDisplacement) {
              chunk_field.displacement(ielement, igll, icomp)[lane] =
                  curr_field.h_field(iglob, icomp);
            }
            if constexpr (StoreVelocity) {
              chunk_field.velocity(ielement, igll, icomp)[lane] =
                  curr_field.h_field_dot(iglob, icomp);
            }
            if constexpr (StoreAcceleration) {
              chunk_field.acceleration(ielement, igll, icomp)[lane] =
                  curr_field.h_field_dot_dot(iglob, icomp);
            }
            if constexpr (StoreMassMatrix) {
              chunk_field.mass_matrix(ielement, igll, icomp)[lane] =
                  curr_field.h_mass_inverse(iglob, icomp);
            }
          }
        }
      }else{
        const int iglob = field.h_assembly_index_mapping(
            field.h_index_mapping(ispec, iz, ix), static_cast<int>(MediumType));

        for (int icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            chunk_field.displacement(ielement, igll, icomp) =
                curr_field.h_field(iglob, icomp);
          }
          if constexpr (StoreVelocity) {
            chunk_field.velocity(ielement, igll, icomp) =
                curr_field.h_field_dot(iglob, icomp);
          }
          if constexpr (StoreAcceleration) {
            chunk_field.acceleration(ielement, igll, icomp);
          }
          if constexpr (StoreMassMatrix) {
            chunk_field.mass_matrix(ielement, igll, icomp) =
                curr_field.h_mass_inverse(iglob, icomp);
          }
        }
      }
    });

  return;
}


}
}
