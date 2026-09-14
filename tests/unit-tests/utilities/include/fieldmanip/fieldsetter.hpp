#pragma once
#include "specfem/assembly.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/element/dimension.hpp"
#include "specfem/execution.hpp"
#include <type_traits>

namespace specfem::test_fieldmanip {

/**
 * @brief Base class for setting fields by pointwise data in set_field_values().
 *
 * `specfem::test_fieldmanip::set_field_values<wavefield_type>(assembly,
 * point_setter)` sets the values of that simfield in device space.
 * Displacement, velocity, and acceleration can be updated, based on if those
 * boolean flags are set.
 *
 * For each degree of freedom `iglob`, point data (such as global coordinates)
 * are given to `point_setter`, which returns a `point::field` result according
 * to that data. The exact behavior of which is modified by overloading the
 * `KOKKOS_INLINE_FUNCTION`s `displacement`, `velocity`, and `acceleration`,
 * respectively.
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct PointSetter {
  static constexpr specfem::element::dimension_tag dimension_tag = DimensionTag;
  static constexpr specfem::element::medium_tag medium_tag = MediumTag;
  static constexpr int ndim = specfem::element::dimension<dimension_tag>::dim;

  bool set_displacement;
  bool set_velocity;
  bool set_acceleration;

  PointSetter(const bool &set_displacement, const bool &set_velocity,
              const bool &set_acceleration)
      : set_displacement(set_displacement), set_velocity(set_velocity),
        set_acceleration(set_acceleration) {}

private:
  using PointTags =
      specfem::tags::Tags<dimension_tag, medium_tag, false /*using_simd*/>;

public:
  using PointDisplacementType = specfem::point::displacement<PointTags>;
  using PointVelocityType = specfem::point::velocity<PointTags>;
  using PointAccelerationType = specfem::point::acceleration<PointTags>;

  struct PointData {
    int iglob;
    specfem::point::global_coordinates<dimension_tag> coords;
    KOKKOS_INLINE_FUNCTION
    PointData(const int &iglob,
              const specfem::point::global_coordinates<dimension_tag> &coords)
        : iglob(iglob), coords(coords) {}
  };

  // DEFAULT SETTERS: INIT TO ZERO
  KOKKOS_INLINE_FUNCTION PointDisplacementType
  displacement(const PointData &data) const {
    PointDisplacementType val;
    constexpr int ncomp =
        specfem::element::attributes<dimension_tag, medium_tag>::components;
    for (int icomp = 0; icomp < ncomp; icomp++) {
      val(icomp) = 0;
    }
    return val;
  }
  KOKKOS_INLINE_FUNCTION PointVelocityType
  velocity(const PointData &data) const {
    PointVelocityType val;
    constexpr int ncomp =
        specfem::element::attributes<dimension_tag, medium_tag>::components;
    for (int icomp = 0; icomp < ncomp; icomp++) {
      val(icomp) = 0;
    }
    return val;
  }
  KOKKOS_INLINE_FUNCTION PointAccelerationType
  acceleration(const PointData &data) const {
    PointAccelerationType val;
    constexpr int ncomp =
        specfem::element::attributes<dimension_tag, medium_tag>::components;
    for (int icomp = 0; icomp < ncomp; icomp++) {
      val(icomp) = 0;
    }
    return val;
  }
};

/**
 * @brief Sets field values (by iglob) of a simfield using a PointSetter.
 */
template <specfem::simulation::field_type field_type, typename PointSetterType,
          specfem::element::dimension_tag dimension_tag>
void set_field_values(
    const specfem::assembly::assembly<dimension_tag> &assembly,
    const PointSetterType &point_setter)
  requires(
      std::is_base_of_v<PointSetter<dimension_tag, PointSetterType::medium_tag>,
                        PointSetterType>)
{
  constexpr auto medium_tag = PointSetterType::medium_tag;
  static_assert(
      std::is_base_of_v<PointSetter<dimension_tag, medium_tag>,
                        PointSetterType>,
      "set_field_values() second argument needs to inherit from a PointSetter");
  const auto field =
      assembly.fields.template get_simulation_field<field_type>();
  const int nglob = field.template get_nglob<medium_tag>();

  if (nglob <= 0) {
    return;
  }
  constexpr bool using_simd = false;
  using PointTags = specfem::tags::Tags<dimension_tag, medium_tag, using_simd>;

  using PointDisplacementType = specfem::point::displacement<PointTags>;
  using PointVelocityType = specfem::point::velocity<PointTags>;
  using PointAccelerationType = specfem::point::acceleration<PointTags>;

  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_configuration::default_range_config<
      simd, Kokkos::DefaultExecutionSpace>;

  // use coords struct so that this works for both dim2 and dim3
  Kokkos::View<specfem::point::global_coordinates<dimension_tag> *> dof_coords(
      "dof_coords", nglob);

  const auto elements =
      assembly.element_types.get_elements_on_device(medium_tag);
  const auto &element_grid = assembly.mesh.element_grid;
  specfem::execution::ChunkedDomainIterator chunk(parallel_config(), elements,
                                                  element_grid);
  specfem::execution::for_all(
      "specfem::nonconforming_test::kernel::acoustic_elastic3d::set_field_"
      "values load dof_coords",
      chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        specfem::point::global_coordinates<dimension_tag> coords;
        specfem::assembly::load_on_device(index, assembly.mesh, coords);
        const int iglob = field.template get_iglob<true, medium_tag>(index);
        dof_coords(iglob) = coords;
      });

  Kokkos::fence();
  specfem::execution::RangeIterator dof_range(parallel_config(), nglob);
  specfem::execution::for_all(
      "specfem::nonconforming_test::kernel::acoustic_elastic3d::set_field_"
      "values valset",
      dof_range,
      KOKKOS_LAMBDA(
          const typename decltype(dof_range)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();

        const typename PointSetterType::PointData point_data(
            index.iglob, dof_coords(index.iglob));

        if (point_setter.set_displacement) {
          PointDisplacementType fieldval =
              point_setter.displacement(point_data);
          specfem::assembly::store_on_device(index, field, fieldval);
        }
        if (point_setter.set_velocity) {
          PointVelocityType fieldval = point_setter.velocity(point_data);
          specfem::assembly::store_on_device(index, field, fieldval);
        }
        if (point_setter.set_acceleration) {
          PointAccelerationType fieldval =
              point_setter.acceleration(point_data);
          specfem::assembly::store_on_device(index, field, fieldval);
        }
      });
  Kokkos::fence();
}
} // namespace specfem::test_fieldmanip
