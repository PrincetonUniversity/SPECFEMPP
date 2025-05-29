#pragma once

#include "compute/assembly/assembly.hpp"
#include "parallel_configuration/range_config.hpp"
#include "policies/range_iterator.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::element::medium_tag MediumTag>
void specfem::kokkos_kernels::impl::invert_mass_matrix(
    const specfem::compute::assembly &assembly) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto wavefield = WavefieldType;
  constexpr auto dimension = DimensionTag;
  const auto field = assembly.fields.get_simulation_field<wavefield>();

  const int nglob = field.template get_nglob<MediumTag>();
  constexpr bool using_simd = true;
  using PointFieldType = specfem::point::field<DimensionTag, MediumTag, false,
                                               false, false, true, using_simd>;

  using ParallelConfig = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  using RangeIterator = specfem::policy::RangeIterator<ParallelConfig>;

  using IndexType = specfem::point::assembly_index<using_simd>;

  RangeIterator range(nglob);

  specfem::execution::for_all(
      "specfem::domain::domain::divide_mass_matrix", range,
      KOKKOS_LAMBDA(const IndexType &index) {
        PointFieldType load_field;
        specfem::compute::load_on_device(index, field, load_field);
        PointFieldType store_field(load_field.invert_mass_matrix());
        specfem::compute::store_on_device(index, store_field, field);
      });

  // Kokkos::fence();
}
