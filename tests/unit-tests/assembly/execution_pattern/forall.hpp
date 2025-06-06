#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "policies/chunk.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace execution_pattern {

template <typename simd, typename ExecutionSpace>
using ParallelConfig = specfem::parallel_config::default_chunk_config<
    specfem::dimension::type::dim2, simd, ExecutionSpace>;

template <typename ViewType, bool using_simd>
using PolicyType = specfem::policy::element_chunk<
    ParallelConfig<specfem::datatype::simd<type_real, using_simd>,
                   typename ViewType::execution_space> >;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <bool using_simd, typename ViewType, typename ClosureType>
inline std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                       Kokkos::DefaultExecutionSpace>,
                        void>
forall(const std::string name, const ViewType elements, int ngll,
       const ClosureType closure) {

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  // Create a chunk policy to iterate over the elements
  using DevicePolicyType = execution_pattern::PolicyType<ViewType, using_simd>;
  constexpr auto simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();
  DevicePolicyType policy(elements, ngll, ngll);

  Kokkos::parallel_for(
      name, static_cast<const typename DevicePolicyType::policy_type &>(policy),
      KOKKOS_LAMBDA(
          const typename PolicyType<ViewType, using_simd>::member_type &team) {
        for (int tile = 0; tile < DevicePolicyType::tile_size * simd_size;
             tile += DevicePolicyType::chunk_size * simd_size) {
          const typename DevicePolicyType::iterator_type iterator =
              policy.league_iterator(team.league_rank(), tile);
          if (iterator.is_end()) {
            return; // No elements to process in this tile
          }

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [=](const int i) {
                const typename PolicyType<ViewType,
                                          using_simd>::iterator_type::index_type
                    iterator_index = iterator(i);
                closure(iterator_index);
              });
        }
      });
}
#endif

template <bool using_simd, typename ViewType, typename ClosureType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultHostExecutionSpace>,
                 void>
forall(const std::string name, const ViewType elements, int ngll,
       const ClosureType closure) {

  using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

  // Create a chunk policy to iterate over the elements
  constexpr auto dimension = specfem::dimension::type::dim2;
  using simd = specfem::datatype::simd<type_real, using_simd>;
  constexpr int simd_size = simd::size();
  using ParallelConfig =
      specfem::parallel_config::default_chunk_config<dimension, simd,
                                                     ExecutionSpace>;
  using HostPolicyType = specfem::policy::element_chunk<ParallelConfig>;
  HostPolicyType policy(elements, ngll, ngll);

  Kokkos::parallel_for(
      name, static_cast<const typename HostPolicyType::policy_type &>(policy),
      [=](const typename HostPolicyType::member_type &team) {
        for (int tile = 0; tile < HostPolicyType::tile_size * simd_size;
             tile += HostPolicyType::chunk_size * simd_size) {
          const typename HostPolicyType::iterator_type iterator =
              policy.league_iterator(team.league_rank(), tile);
          if (iterator.is_end()) {
            return; // No elements to process in this tile
          }

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [=](const int i) {
                const typename PolicyType<ViewType,
                                          using_simd>::iterator_type::index_type
                    iterator_index = iterator(i);
                closure(iterator_index);
              });
        }
      });
}
} // namespace execution_pattern
