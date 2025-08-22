#pragma once

#include "execution/for_all.hpp"
#include "execution/range_iterator.hpp"
#include "parallel_configuration/range_config.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::element::medium_tag MediumTag>
void specfem::kokkos_kernels::impl::divide_mass_matrix(
    const specfem::assembly::assembly<DimensionTag> &assembly) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto wavefield = WavefieldType;
  constexpr auto dimension = DimensionTag;
  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  const int nglob = field.template get_nglob<medium_tag>();

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif

  using PointAccelerationType =
      specfem::point::acceleration<dimension, medium_tag, using_simd>;
  using PointMassInverseType =
      specfem::point::mass_inverse<dimension, medium_tag, using_simd>;

  using parallel_config = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  using IndexType = specfem::point::assembly_index<using_simd>;

  specfem::execution::RangeIterator range(parallel_config(), nglob);

  specfem::execution::for_all(
      "specfem::kokkos_kernels::divide_mass_matrix", range,
      KOKKOS_LAMBDA(const IndexType &index) {
        PointAccelerationType acceleration;
        PointMassInverseType mass_inverse;
        specfem::assembly::load_on_device(index, field, acceleration,
                                          mass_inverse);
        for (int icomp = 0; icomp < PointAccelerationType::components;
             ++icomp) {
          acceleration(icomp) *= mass_inverse(icomp);
        }
        specfem::assembly::store_on_device(index, field, acceleration);
      });

  // Kokkos::fence();

  return;
}
