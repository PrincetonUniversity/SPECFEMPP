#pragma once

#include "enumerations/interface.hpp"
#include "medium/medium.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/assembly/impl/helper.hpp"
#include "specfem/chunk_element.hpp"
#include "specfem/execution.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::assembly_impl {

/**
 * @brief Template metaprogramming utility for multi-dimensional array type
 * generation
 *
 * Recursively builds pointer types for multi-dimensional arrays used in source
 * data storage. This helper enables compile-time generation of appropriate
 * Kokkos view types for different dimensional problems.
 *
 * @tparam T Base data type (typically `type_real`)
 * @tparam Rank Number of array dimensions to generate
 */
template <typename T, int Rank> struct ExtentImpl {
  /**
   * @brief Recursive type definition for multi-dimensional pointer
   */
  using type = typename ExtentImpl<T, Rank - 1>::type *;
};

/**
 * @brief Base case specialization for rank-0 arrays
 *
 * @tparam T Base data type
 */
template <typename T> struct ExtentImpl<T, 0> {
  /**
   * @brief Base type for rank-0 case
   */
  using type = T;
};

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
class helper {
public:
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;
  constexpr static auto ngll = NGLL;
  constexpr static bool using_simd = false;

  /**
   * @brief Number of spatial dimensions (2 for 2D, 3 for 3D)
   */
  constexpr static auto ndim =
      (dimension_tag == specfem::dimension::type::dim2) ? 2 : 3;

  /**
   * @brief Rank of source array (4 for 2D: [sources][components][z][x],
   *        5 for 3D: [sources][components][z][y][x])
   */
  constexpr static int wavefield_grid_rank = ndim + 2;
  ///@}

  // Create dimension-dependent viewtype for wavefield on entire gridusing
  // SourceArrayView =
  using WavefieldOnEntireGridViewType =
      Kokkos::View<typename ExtentImpl<type_real, wavefield_grid_rank>::type,
                   Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  // Constructor
  helper(specfem::assembly::assembly<dimension_tag> assembly,
         WavefieldOnEntireGridViewType wavefield_on_entire_grid)
      : assembly(assembly), wavefield_on_entire_grid(wavefield_on_entire_grid) {
    const auto &element_grid = assembly.mesh.element_grid;
    if (element_grid != ngll) {
      throw std::runtime_error("Number of quadrature points not supported");
    }
  };

  void operator()(const specfem::wavefield::type wavefield_type) {
    const auto buffer = assembly.fields.buffer;

    // Get the element grid (ngllx, nglly, ngllz)
    const auto &element_grid = assembly.mesh.element_grid;

    const auto elements =
        assembly.element_types.get_elements_on_device(medium_tag, property_tag);

    const int nelements = elements.extent(0);

    if (nelements == 0) {
      return;
    }

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    constexpr int nthreads = 32;
    constexpr int lane_size = 1;
    constexpr int chunk_size =
        (dimension_tag == specfem::dimension::type::dim2) ? 16 : 4;
#else
    constexpr int nthreads = 1;
    constexpr int lane_size = 1;
    constexpr int chunk_size = 1;
#endif

    using no_simd = specfem::datatype::simd<type_real, false>;

    using ParallelConfig = specfem::parallel_configuration::chunk_config<
        dimension_tag, chunk_size, 1, nthreads, lane_size, no_simd,
        Kokkos::DefaultExecutionSpace>;

    using ChunkDisplacementType =
        specfem::chunk_element::displacement<ParallelConfig::chunk_size, ngll,
                                             dimension_tag, medium_tag,
                                             using_simd>;
    using ChunkVelocityType =
        specfem::chunk_element::velocity<ParallelConfig::chunk_size, ngll,
                                         dimension_tag, medium_tag, using_simd>;
    using ChunkAccelerationType =
        specfem::chunk_element::acceleration<ParallelConfig::chunk_size, ngll,
                                             dimension_tag, medium_tag,
                                             using_simd>;

    using QuadratureType = specfem::quadrature::lagrange_derivative<
        ngll, dimension_tag, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    using PointPropertyType =
        specfem::point::properties<dimension_tag, medium_tag, property_tag,
                                   false>;

    using PointFieldDerivativesType =
        specfem::point::field_derivatives<dimension_tag, medium_tag, false>;

    int scratch_size =
        ChunkDisplacementType::shmem_size() + ChunkVelocityType::shmem_size() +
        ChunkAccelerationType::shmem_size() + QuadratureType::shmem_size();

    specfem::execution::ChunkedDomainIterator chunk(ParallelConfig(), elements,
                                                    element_grid);

    specfem::execution::for_each_level(
        "specfem::assembly::assembly::compute_wavefield",
        chunk.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_CLASS_LAMBDA(
            const typename decltype(chunk)::index_type chunk_iterator_index) {
          const auto &chunk_index = chunk_iterator_index.get_index();
          const auto team = chunk_index.get_policy_index();
          QuadratureType lagrange_derivative(team);
          ChunkDisplacementType displacement(team.team_scratch(0));
          ChunkVelocityType velocity(team.team_scratch(0));
          ChunkAccelerationType acceleration(team.team_scratch(0));

          specfem::assembly::load_on_device(team, assembly.mesh,
                                            lagrange_derivative);

          specfem::assembly::load_on_device(chunk_index, buffer, displacement,
                                            velocity, acceleration);
          team.team_barrier();

          // Get the wavefield subview based on dimension
          const auto wavefield = [&]() {
            if constexpr (dimension_tag == specfem::dimension::type::dim3) {
              return Kokkos::subview(wavefield_on_entire_grid,
                                     chunk_index.get_range(), Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
            } else {
              return Kokkos::subview(wavefield_on_entire_grid,
                                     chunk_index.get_range(), Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
            }
          }();

          // Call the compute_wavefield function
          specfem::medium::compute_wavefield<dimension_tag, MediumTag,
                                             PropertyTag>(
              chunk_index, assembly, lagrange_derivative, displacement,
              velocity, acceleration, wavefield_type, wavefield);
        });

    return;
  }

private:
  const specfem::assembly::assembly<dimension_tag> assembly;
  WavefieldOnEntireGridViewType wavefield_on_entire_grid;
};

} // namespace specfem::assembly::assembly_impl
