#pragma once

#include "domain.hpp"
#include "parallel_configuration/range_config.hpp"
#include "policies/range.hpp"
#include <Kokkos_Core.hpp>

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename qp_type>
void specfem::domain::domain<WavefieldType, DimensionType, MediumTag,
                             qp_type>::divide_mass_matrix() {
  const int nglob = field.template get_nglob<MediumTag>();
  constexpr bool using_simd = true;
  using LoadFieldType = specfem::point::field<DimensionType, MediumTag, false,
                                              false, true, true, using_simd>;
  using StoreFieldType = specfem::point::field<DimensionType, MediumTag, false,
                                               false, true, false, using_simd>;

  using ParallelConfig = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  using RangePolicy = specfem::policy::range<ParallelConfig>;

  RangePolicy range(nglob);

  Kokkos::parallel_for(
      "specfem::domain::domain::divide_mass_matrix",
      static_cast<typename RangePolicy::policy_type &>(range),
      KOKKOS_CLASS_LAMBDA(const int iglob) {
        const auto iterator = range.range_iterator(iglob);
        const auto index = iterator(0);

        LoadFieldType load_field;
        specfem::compute::load_on_device(index.index, field, load_field);
        StoreFieldType store_field(load_field.divide_mass_matrix());
        specfem::compute::store_on_device(index.index, store_field, field);
      });

  Kokkos::fence();

  return;
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename qp_type>
void specfem::domain::domain<WavefieldType, DimensionType, MediumTag,
                             qp_type>::invert_mass_matrix() {
  const int nglob = field.template get_nglob<MediumTag>();
  constexpr bool using_simd = true;
  using PointFieldType = specfem::point::field<DimensionType, MediumTag, false,
                                               false, false, true, using_simd>;

  using ParallelConfig = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  using RangePolicy = specfem::policy::range<ParallelConfig>;

  RangePolicy range(nglob);

  Kokkos::parallel_for(
      "specfem::domain::domain::divide_mass_matrix",
      static_cast<typename RangePolicy::policy_type &>(range),
      KOKKOS_CLASS_LAMBDA(const int iglob) {
        const auto iterator = range.range_iterator(iglob);
        const auto index = iterator(0);

        PointFieldType load_field;
        specfem::compute::load_on_device(index.index, field, load_field);
        PointFieldType store_field(load_field.invert_mass_matrix());
        specfem::compute::store_on_device(index.index, store_field, field);
      });

  Kokkos::fence();
}
