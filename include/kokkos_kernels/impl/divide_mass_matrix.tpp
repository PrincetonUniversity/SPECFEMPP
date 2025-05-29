#pragma once

#include "compute/assembly/assembly.hpp"
#include "parallel_configuration/range_config.hpp"
#include "execution/range_iterator.hpp"
#include "execution/for_all.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::element::medium_tag MediumTag>
void specfem::kokkos_kernels::impl::divide_mass_matrix(
    const specfem::compute::assembly &assembly) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto wavefield = WavefieldType;
  constexpr auto dimension = DimensionTag;
  const auto field = assembly.fields.get_simulation_field<wavefield>();

  const int nglob = field.template get_nglob<MediumTag>();
  constexpr bool using_simd = true;
  using LoadFieldType = specfem::point::field<DimensionTag, MediumTag, false,
                                              false, true, true, using_simd>;
  using StoreFieldType = specfem::point::field<DimensionTag, MediumTag, false,
                                               false, true, false, using_simd>;

  using parallel_config = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  using IndexType = specfem::point::assembly_index<using_simd>;

  specfem::execution::RangeIterator range(parallel_config(), nglob);

  specfem::execution::for_all(
      "specfem::domain::domain::divide_mass_matrix", range,
      KOKKOS_LAMBDA(const IndexType &index) {
        LoadFieldType load_field;
        specfem::compute::load_on_device(index, field, load_field);
        StoreFieldType store_field(load_field.divide_mass_matrix());
        specfem::compute::store_on_device(index, store_field, field);
      });

  // Kokkos::fence();

  return;
}
