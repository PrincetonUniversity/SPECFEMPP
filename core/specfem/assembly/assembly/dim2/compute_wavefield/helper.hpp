#pragma once

#include "enumerations/interface.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_each_level.hpp"
#include "medium/medium.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/chunk_element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace impl {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
class helper {
public:
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;
  constexpr static auto ngll = NGLL;
  constexpr static bool using_simd = false;

  helper(specfem::assembly::assembly<specfem::dimension::type::dim2> assembly,
         Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                      Kokkos::DefaultExecutionSpace>
             wavefield_on_entire_grid)
      : assembly(assembly), wavefield_on_entire_grid(wavefield_on_entire_grid) {
    if (assembly.mesh.ngllz != ngll || assembly.mesh.ngllx != ngll) {
      throw std::runtime_error("Number of quadrature points not supported");
    }
  }

  void operator()(const specfem::wavefield::type wavefield_type) {
    const auto buffer = assembly.fields.buffer;

    const int ngllz = assembly.mesh.ngllz;
    const int ngllx = assembly.mesh.ngllx;

    const auto elements =
        assembly.element_types.get_elements_on_device(medium_tag, property_tag);

    const int nelements = elements.extent(0);

    if (nelements == 0) {
      return;
    }

    using ParallelConfig = specfem::parallel_config::default_chunk_config<
        specfem::dimension::type::dim2,
        specfem::datatype::simd<type_real, false>,
        Kokkos::DefaultExecutionSpace>;

    using ChunkDisplacementType = specfem::chunk_element::displacement<
        specfem::parallel_config::chunk_size, ngll, dimension, medium_tag,
        using_simd>;
    using ChunkVelocityType =
        specfem::chunk_element::velocity<specfem::parallel_config::chunk_size,
                                         ngll, dimension, medium_tag,
                                         using_simd>;
    using ChunkAccelerationType = specfem::chunk_element::acceleration<
        specfem::parallel_config::chunk_size, ngll, dimension, medium_tag,
        using_simd>;

    using QuadratureType = specfem::element::quadrature<
        ngll, specfem::dimension::type::dim2, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

    using PointPropertyType =
        specfem::point::properties<specfem::dimension::type::dim2, medium_tag,
                                   property_tag, false>;

    using PointFieldDerivativesType =
        specfem::point::field_derivatives<specfem::dimension::type::dim2,
                                          medium_tag, false>;

    int scratch_size =
        ChunkDisplacementType::shmem_size() + ChunkVelocityType::shmem_size() +
        ChunkAccelerationType::shmem_size() + QuadratureType::shmem_size();

    specfem::execution::ChunkedDomainIterator chunk(ParallelConfig(), elements,
                                                    ngllz, ngllx);

    specfem::execution::for_each_level(
        "specfem::assembly::assembly::compute_wavefield",
        chunk.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_CLASS_LAMBDA(
            const typename decltype(chunk)::index_type chunk_iterator_index) {
          const auto &chunk_index = chunk_iterator_index.get_index();
          const auto team = chunk_index.get_policy_index();
          QuadratureType quadrature(team);
          ChunkDisplacementType displacement(team.team_scratch(0));
          ChunkVelocityType velocity(team.team_scratch(0));
          ChunkAccelerationType acceleration(team.team_scratch(0));

          specfem::assembly::load_on_device(team, assembly.mesh, quadrature);

          specfem::assembly::load_on_device(chunk_index, buffer, displacement,
                                            velocity, acceleration);
          team.team_barrier();

          const auto wavefield =
              Kokkos::subview(wavefield_on_entire_grid, chunk_index.get_range(),
                              Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

          specfem::medium::compute_wavefield<MediumTag, PropertyTag>(
              chunk_index, assembly, quadrature, displacement, velocity,
              acceleration, wavefield_type, wavefield);
        });

    return;
  }

private:
  const specfem::assembly::assembly<specfem::dimension::type::dim2> assembly;
  Kokkos::View<type_real ****, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      wavefield_on_entire_grid;
};

} // namespace impl
