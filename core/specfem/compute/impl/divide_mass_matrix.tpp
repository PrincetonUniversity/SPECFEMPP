#pragma once

#include "specfem/execution.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/point.hpp"
#include "divide_mass_matrix.hpp"
#include "specfem/macros.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::compute::impl {

template <typename Tags>
void divide_mass_matrix_core(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {

  constexpr auto medium_tag = Tags::medium_tag;
  constexpr auto wavefield = Tags::wavefield_tag;
  constexpr auto dimension_tag = Tags::dimension_tag;
  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  const int nglob = field.template get_nglob<medium_tag>();

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  // TODO(Rohit : DIM3_SIMD) Enable simd execution for dim3 solver
  constexpr bool using_simd = (dimension_tag == specfem::element::dimension_tag::dim2) ? true : false;
#endif

  using simd = specfem::datatype::simd<type_real, using_simd>;

  using PointTags = specfem::tags::Tags<dimension_tag, medium_tag, using_simd>;

  using PointAccelerationType =
      specfem::point::acceleration<PointTags>;
  using PointMassInverseType =
      specfem::point::mass_inverse<PointTags>;

  using parallel_config = specfem::parallel_configuration::default_range_config<
      simd,
      Kokkos::DefaultExecutionSpace>;

  using IndexType = specfem::point::assembly_index<using_simd>;

  specfem::execution::RangeIterator range(parallel_config(), nglob);

  Kokkos::Profiling::pushRegion("Divide Mass Matrix");

  specfem::execution::for_all(
      "specfem::compute::divide_mass_matrix", range,
      KOKKOS_LAMBDA(const typename decltype(range)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
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

  Kokkos::Profiling::popRegion();

  // Kokkos::fence();

  return;
}

template <int NGLL, typename Tags>
void divide_mass_matrix(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {

  constexpr auto WavefieldType = Tags::wavefield_tag;
  constexpr auto DimensionTag = Tags::dimension_tag;
  constexpr auto MediumTag = Tags::medium_tag;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2, DIM3),
       MEDIUM_TAG(ELASTIC, ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T, ELASTIC_SPIN)),
      {
        if constexpr (DimensionTag == _dimension_tag_ &&
                      MediumTag == _medium_tag_) {
          divide_mass_matrix_core<
              specfem::tags::Tags<DimensionTag, WavefieldType, _medium_tag_> >(
              assembly);
        }
      })
}
} // namespace specfem::compute::impl
