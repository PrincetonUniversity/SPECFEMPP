#pragma once
#include "specfem/assembly.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/execution.hpp"
#include <type_traits>

namespace specfem::test_fieldmanip {

/**
 * @brief Recovers the field values (by iglob) of a simfield from assembly.
 *
 * The resulting view is in device space.
 */
template <specfem::simulation::field_type field_type,
          specfem::element::dimension_tag dimension_tag,
          specfem::element::medium_tag medium_tag,
          specfem::data_access::DataClassType ClassToGet>
Kokkos::View<type_real *[specfem::element::attributes<dimension_tag,
                                                      medium_tag>::components]>
get_field_values(const specfem::assembly::assembly<dimension_tag> &assembly) {
  static_assert(
      ClassToGet == specfem::data_access::DataClassType::displacement ||
          ClassToGet == specfem::data_access::DataClassType::velocity ||
          ClassToGet == specfem::data_access::DataClassType::acceleration,
      "get_field_values ClassToGet must be displacement, velocity, or "
      "acceleration");
  const auto field =
      assembly.fields.template get_simulation_field<field_type>();
  const int nglob = field.template get_nglob<medium_tag>();

  constexpr int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;
  Kokkos::View<type_real *[ncomp]> fieldvals("fieldvals", nglob);
  if (nglob <= 0) {
    return fieldvals;
  }
  constexpr bool using_simd = false;
  using PointTags = specfem::tags::Tags<dimension_tag, medium_tag, using_simd>;

  using PointDisplacementType = specfem::point::displacement<PointTags>;
  using PointVelocityType = specfem::point::velocity<PointTags>;
  using PointAccelerationType = specfem::point::acceleration<PointTags>;

  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_configuration::default_range_config<
      simd, Kokkos::DefaultExecutionSpace>;

  using PointAccessorType = std::conditional_t<
      ClassToGet == specfem::data_access::DataClassType::displacement,
      PointDisplacementType,
      std::conditional_t<ClassToGet ==
                             specfem::data_access::DataClassType::velocity,
                         PointVelocityType, PointAccelerationType>>;

  specfem::execution::RangeIterator dof_range(parallel_config(), nglob);
  specfem::execution::for_all(
      "specfem::test_fieldmanip::get_field_values", dof_range,
      KOKKOS_LAMBDA(
          const typename decltype(dof_range)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();

        PointAccessorType fieldval;
        specfem::assembly::load_on_device(index, field, fieldval);
        for (int icomp = 0; icomp < ncomp; icomp++) {
          fieldvals(index.iglob, icomp) = fieldval(icomp);
        }
      });
  Kokkos::fence();
  return fieldvals;
}
} // namespace specfem::test_fieldmanip
