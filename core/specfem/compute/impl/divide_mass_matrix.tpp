#pragma once

#include "divide_mass_matrix.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/execution.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/point.hpp"
#include "specfem/tag_dispatch.hpp"
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
  constexpr bool using_simd = true;
#endif

  using simd = specfem::datatype::simd<type_real, using_simd>;

  using PointTags = specfem::tags::Tags<dimension_tag, medium_tag, using_simd>;

  using PointAccelerationType = specfem::point::acceleration<PointTags>;
  using PointMassInverseType = specfem::point::mass_inverse<PointTags>;

  using parallel_config = specfem::parallel_configuration::default_range_config<
      simd, Kokkos::DefaultExecutionSpace>;

  using IndexType = specfem::point::assembly_index<using_simd>;

  specfem::execution::RangeIterator range(parallel_config(), nglob);

  Kokkos::Profiling::pushRegion("Divide Mass Matrix");

  specfem::execution::for_all(
      "specfem::compute::divide_mass_matrix", range,
      KOKKOS_LAMBDA(
          const typename decltype(range)::base_index_type &iterator_index) {
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
  // Call divide_mass_matrix_core only if Tags represent a valid combination of
  // dimension and medium tags
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          specfem::tag_dispatch::medium_set<Tags::medium_tag>{},
      [&]<typename ElementTags>() { divide_mass_matrix_core<Tags>(assembly); });
}
} // namespace specfem::compute::impl
