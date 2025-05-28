#pragma once

#include "compute/assembly/assembly.hpp"
#include "parallel_configuration/range_config.hpp"
#include "policies/range_iterator.hpp"
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

  using ParallelConfig = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  using RangePolicy = specfem::policy::RangeIterator<ParallelConfig>;

  using IndexType = specfem::point::assembly_index<using_simd>;

  RangePolicy range(nglob);

  specfem::execution::for_all(
      "specfem::domain::domain::divide_mass_matrix", range,
      KOKKOS_LAMBDA(const IndexType &index) {
        LoadFieldType load_field;
        specfem::compute::load_on_device(index, field, load_field);
        StoreFieldType store_field(load_field.divide_mass_matrix());
        specfem::compute::store_on_device(index, store_field, field);
      });

  // specfem::execution::for_each(
  //     "specfem::domain::domain::divide_mass_matrix", range,
  //     KOKKOS_LAMBDA(const typename RangePolicy::index_type &range_index) {
  //       const auto tile_iterator = range_index.get_iterator();
  //       specfem::execution::for_each(
  //           tile_iterator, [=](const typename
  //           decltype(tile_iterator)::index_type &tile_index) {
  //             const auto &index = tile_index.get_index();
  //             LoadFieldType load_field;
  //             specfem::compute::load_on_device(index, field, load_field);
  //             StoreFieldType store_field(load_field.divide_mass_matrix());
  //             specfem::compute::store_on_device(index, store_field, field);
  //           });
  //     });

  // Kokkos::parallel_for(
  //     "specfem::domain::domain::divide_mass_matrix",
  //     static_cast<typename RangePolicy::policy_type &>(range),
  //     KOKKOS_LAMBDA(const int iglob) {
  //       for (int itile = 0; itile < RangePolicy::tile_size; ++itile) {

  //         const auto iterator = range.range_iterator(iglob, itile);

  //         if (iterator.is_end()) {
  //           return; // Skip if the iterator is at the end
  //         }
  //         const auto index = iterator();

  //         LoadFieldType load_field;
  //         specfem::compute::load_on_device(index.index, field, load_field);
  //         StoreFieldType store_field(load_field.divide_mass_matrix());
  //         specfem::compute::store_on_device(index.index, store_field, field);
  //       }
  //     });

  // Kokkos::fence();

  return;
}
