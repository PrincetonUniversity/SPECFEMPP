#pragma once

#include "specfem/execution.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/timescheme/newmark.hpp"

#include <ostream>
#include <sstream>
#include <string>

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::simulation::field_type WavefieldType>
int specfem::time_scheme::newmark_impl::corrector_phase_impl(
    const specfem::assembly::simulation_field<DimensionTag, WavefieldType>
        &field,
    const type_real deltatover2) {

  constexpr int ncomponents =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  const int nglob = field.template get_nglob<MediumTag>();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif
  using PointAccelerationType = specfem::point::acceleration<
      specfem::tags::Tags<DimensionTag, MediumTag, using_simd>>;

  using PointVelocityType = specfem::point::velocity<
      specfem::tags::Tags<DimensionTag, MediumTag, using_simd>>;

  using parallel_config = specfem::parallel_configuration::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  specfem::execution::RangeIterator range(parallel_config(), nglob);

  Kokkos::Profiling::pushRegion("Compute Corrector Phase");

  specfem::execution::for_all(
      "specfem::time_scheme::newmark_impl::corrector_phase_impl", range,
      KOKKOS_LAMBDA(
          const typename decltype(range)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        PointAccelerationType acceleration;
        PointVelocityType velocity;

        specfem::assembly::load_on_device(index, field, velocity, acceleration);

        for (int idim = 0; idim < ncomponents; ++idim) {
          velocity(idim) += deltatover2 * acceleration(idim);
        }

        specfem::assembly::store_on_device(index, field, velocity);
      });

  Kokkos::Profiling::popRegion();

  return nglob * ncomponents;
}

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::simulation::field_type WavefieldType>
int specfem::time_scheme::newmark_impl::predictor_phase_impl(
    const specfem::assembly::simulation_field<DimensionTag, WavefieldType>
        &field,
    const type_real deltat, const type_real deltatover2,
    const type_real deltasquareover2) {

  constexpr int ncomponents =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  const int nglob = field.template get_nglob<MediumTag>();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif

  using PointTags = specfem::tags::Tags<DimensionTag, MediumTag, using_simd>;

  using PointAccelerationType = specfem::point::acceleration<PointTags>;
  using PointVelocityType = specfem::point::velocity<PointTags>;
  using PointDisplacementType = specfem::point::displacement<PointTags>;

  using parallel_config = specfem::parallel_configuration::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  specfem::execution::RangeIterator range(parallel_config(), nglob);

  Kokkos::Profiling::pushRegion("Compute Predictor Phase");

  specfem::execution::for_all(
      "specfem::time_scheme::newmark_impl::predictor_phase_impl", range,
      KOKKOS_LAMBDA(
          const typename decltype(range)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        PointDisplacementType displacement;
        PointVelocityType velocity;
        PointAccelerationType acceleration;

        specfem::assembly::load_on_device(index, field, displacement, velocity,
                                          acceleration);

        for (int idim = 0; idim < ncomponents; ++idim) {
          displacement(idim) +=
              deltat * velocity(idim) + deltasquareover2 * acceleration(idim);

          velocity(idim) += deltatover2 * acceleration(idim);
          acceleration(idim) = 0;
        }

        specfem::assembly::store_on_device(index, field, displacement, velocity,
                                           acceleration);
      });

  Kokkos::Profiling::popRegion();

  return nglob * ncomponents;
}

namespace specfem::time_scheme::newmark_impl {

template <typename AssemblyFields,
          specfem::simulation::field_type WavefieldType>
int apply_corrector_phase(
    const specfem::assembly::simulation_field<AssemblyFields::dimension_tag,
                                              WavefieldType> &field,
    const specfem::element::medium_tag tag, const type_real deltatover2) {

  int result = 0;
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<AssemblyFields::dimension_tag>{} *
          MEDIUM_SET(elastic, elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t),
      [&]<typename ElementTags>() {
        if (tag == ElementTags::medium_tag) {
          result = specfem::time_scheme::newmark_impl::corrector_phase_impl<
              AssemblyFields::dimension_tag, ElementTags::medium_tag,
              WavefieldType>(field, deltatover2);
        }
      });

  return result;
}

template <typename AssemblyFields,
          specfem::simulation::field_type WavefieldType>
int apply_predictor_phase(
    const specfem::assembly::simulation_field<AssemblyFields::dimension_tag,
                                              WavefieldType> &field,
    const specfem::element::medium_tag tag, const type_real deltat,
    const type_real deltatover2, const type_real deltasquareover2) {

  int result = 0;
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<AssemblyFields::dimension_tag>{} *
          MEDIUM_SET(elastic, elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t),
      [&]<typename ElementTags>() {
        if (tag == ElementTags::medium_tag) {
          result = specfem::time_scheme::newmark_impl::predictor_phase_impl<
              AssemblyFields::dimension_tag, ElementTags::medium_tag,
              WavefieldType>(field, deltat, deltatover2, deltasquareover2);
        }
      });

  return result;
}

std::string to_string(const std::string &simulation_type,
                      const type_real deltat, const type_real t0) {
  std::ostringstream message;
  message << "  Time Scheme:\n"
          << "------------------------------\n"
          << "- Newmark\n"
          << "    simulation type = " << simulation_type << "\n"
          << "    dt = " << deltat << "\n"
          << "    Start time = " << t0 << "\n";

  return message.str();
}

} // namespace specfem::time_scheme::newmark_impl

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::forward>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_corrector_phase<
      AssemblyFields, specfem::simulation::field_type::forward>(
      fields.forward, tag, deltatover2);
}

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::forward>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_predictor_phase<
      AssemblyFields, specfem::simulation::field_type::forward>(
      fields.forward, tag, deltat, deltatover2, deltasquareover2);
}

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::combined>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_corrector_phase<
      AssemblyFields, specfem::simulation::field_type::adjoint>(
      fields.adjoint, tag, deltatover2);
}

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::combined>::
    apply_corrector_phase_backward(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_corrector_phase<
      AssemblyFields, specfem::simulation::field_type::backward>(
      fields.backward, tag, -1.0 * deltatover2);
}

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::combined>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_predictor_phase<
      AssemblyFields, specfem::simulation::field_type::adjoint>(
      fields.adjoint, tag, deltat, deltatover2, deltasquareover2);
}

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::combined>::
    apply_predictor_phase_backward(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_predictor_phase<
      AssemblyFields, specfem::simulation::field_type::backward>(
      fields.backward, tag, -1.0 * deltat, -1.0 * deltatover2,
      deltasquareover2);
}

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::combined_undoatt>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_corrector_phase<
      AssemblyFields, specfem::simulation::field_type::forward>(
      fields.forward, tag, deltatover2);
}

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::combined_undoatt>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_predictor_phase<
      AssemblyFields, specfem::simulation::field_type::forward>(
      fields.forward, tag, deltat, deltatover2, deltasquareover2);
}

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::combined_undoatt>::
    apply_corrector_phase_adjoint(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_corrector_phase<
      AssemblyFields, specfem::simulation::field_type::adjoint>(
      fields.adjoint, tag, deltatover2);
}

template <typename AssemblyFields>
int specfem::time_scheme::newmark<AssemblyFields,
                                  specfem::simulation::type::combined_undoatt>::
    apply_predictor_phase_adjoint(const specfem::element::medium_tag tag) {

  return specfem::time_scheme::newmark_impl::apply_predictor_phase<
      AssemblyFields, specfem::simulation::field_type::adjoint>(
      fields.adjoint, tag, deltat, deltatover2, deltasquareover2);
}

template <typename AssemblyFields>
std::string specfem::time_scheme::newmark<
    AssemblyFields, specfem::simulation::type::forward>::to_string() const {
  return specfem::time_scheme::newmark_impl::to_string("forward", this->deltat,
                                                       this->t0);
}

template <typename AssemblyFields>
void specfem::time_scheme::newmark<
    AssemblyFields,
    specfem::simulation::type::forward>::print(std::ostream &message) const {
  message << this->to_string();
}

template <typename AssemblyFields>
std::string specfem::time_scheme::newmark<
    AssemblyFields, specfem::simulation::type::combined>::to_string() const {
  return specfem::time_scheme::newmark_impl::to_string("adjoint", this->deltat,
                                                       this->t0);
}

template <typename AssemblyFields>
void specfem::time_scheme::newmark<
    AssemblyFields,
    specfem::simulation::type::combined>::print(std::ostream &message) const {
  message << this->to_string();
}

template <typename AssemblyFields>
std::string specfem::time_scheme::newmark<
    AssemblyFields, specfem::simulation::type::combined_undoatt>::to_string()
    const {
  return specfem::time_scheme::newmark_impl::to_string("combined_undoatt",
                                                       this->deltat, this->t0);
}

template <typename AssemblyFields>
void specfem::time_scheme::
    newmark<AssemblyFields, specfem::simulation::type::combined_undoatt>::print(
        std::ostream &message) const {
  message << this->to_string();
}
