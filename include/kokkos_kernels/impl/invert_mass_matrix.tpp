#pragma once

#include "specfem/assembly.hpp"
#include "parallel_configuration/range_config.hpp"
#include "execution/range_iterator.hpp"
#include "execution/for_all.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::element::medium_tag MediumTag>
void specfem::kokkos_kernels::impl::invert_mass_matrix(
    const specfem::assembly::assembly<DimensionTag> &assembly) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto wavefield = WavefieldType;
  constexpr auto dimension = DimensionTag;
  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  const int nglob = field.template get_nglob<medium_tag>();

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  // TODO(Rohit : DIM3_SIMD) Enable simd execution for dim3 solver
  constexpr bool using_simd = (DimensionTag == specfem::dimension::type::dim2) ? true : false;
#endif

  using PointMassType =
      specfem::point::mass_inverse<dimension, medium_tag, using_simd>;

  using parallel_config = specfem::parallel_configuration::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  using IndexType = specfem::point::assembly_index<using_simd>;

  specfem::execution::RangeIterator range(parallel_config(), nglob);

  specfem::execution::for_all(
      "specfem::kokkos_kernels::divide_mass_matrix", range,
      KOKKOS_LAMBDA(const typename decltype(range)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        PointMassType mass;
        specfem::assembly::load_on_device(index, field, mass);
        for (int icomp = 0; icomp < PointMassType::components;
             ++icomp) {
          mass(icomp) = static_cast<type_real>(1.0) / mass(icomp);
        }
        specfem::assembly::store_on_device(index, field, mass);
      });

  // Kokkos::fence();
}
