#ifndef _SPECFEM_TIMESCHEME_NEWMARK_TPP_
#define _SPECFEM_TIMESCHEME_NEWMARK_TPP_

#include "parallel_configuration/range_config.hpp"
#include "policies/range.hpp"
#include "timescheme/newmark.hpp"

namespace {
template <specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
int corrector_phase_impl(
    const specfem::compute::simulation_field<WavefieldType> &field,
    const type_real deltatover2) {

  constexpr int components =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   MediumTag>::components();
  const int nglob = field.template get_nglob<MediumTag>();
  constexpr bool using_simd = true;
  using LoadFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumTag, false,
                            false, true, false, using_simd>;
  using AddFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumTag, false,
                            true, false, false, using_simd>;

  using ParallelConfig = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  using RangePolicyType = specfem::policy::range<ParallelConfig>;

  RangePolicyType range_policy(nglob);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::corrector_phase_impl",
      static_cast<typename RangePolicyType::policy_type &>(range_policy),
      KOKKOS_LAMBDA(const int iglob) {
        const auto iterator = range_policy.range_iterator(iglob);
        const auto index = iterator(0);

        LoadFieldType load;
        AddFieldType add;

        specfem::compute::load_on_device(index.index, field, load);

        for (int idim = 0; idim < components; ++idim) {
          add.velocity(idim) += deltatover2 * load.acceleration(idim);
        }

        specfem::compute::add_on_device(index.index, add, field);
      });

  return nglob * specfem::element::attributes<specfem::dimension::type::dim2,
                                              MediumTag>::components();
}

template <specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
int predictor_phase_impl(
    const specfem::compute::simulation_field<WavefieldType> &field,
    const type_real deltat, const type_real deltatover2,
    const type_real deltasquareover2) {

  constexpr int components =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   MediumTag>::components();
  const int nglob = field.template get_nglob<MediumTag>();
  constexpr bool using_simd = true;
  using LoadFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumTag, false,
                            true, true, false, using_simd>;
  using AddFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumTag, true,
                            true, false, false, using_simd>;
  using StoreFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumTag, false,
                            false, true, false, using_simd>;

  using ParallelConfig = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  using RangePolicyType = specfem::policy::range<ParallelConfig>;

  RangePolicyType range_policy(nglob);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::predictor_phase_impl",
      static_cast<typename RangePolicyType::policy_type &>(range_policy),
      KOKKOS_LAMBDA(const int iglob) {
        const auto iterator = range_policy.range_iterator(iglob);
        const auto index = iterator(0);

        LoadFieldType load;
        AddFieldType add;
        StoreFieldType store;

        specfem::compute::load_on_device(index.index, field, load);

        for (int idim = 0; idim < components; ++idim) {
          add.displacement(idim) += deltat * load.velocity(idim) +
                                    deltasquareover2 * load.acceleration(idim);

          add.velocity(idim) += deltatover2 * load.acceleration(idim);

          store.acceleration(idim) = 0;
        }

        specfem::compute::add_on_device(index.index, add, field);
        specfem::compute::store_on_device(index.index, store, field);
      });

  return nglob * specfem::element::attributes<specfem::dimension::type::dim2,
                                              MediumTag>::components();
}
} // namespace

int specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::forward;

#define APPLY_CORRECTOR_PHASE(DIMENSION_TAG, MEDIUM_TAG)                       \
  if (tag == GET_TAG(MEDIUM_TAG)) {                                            \
    return corrector_phase_impl<GET_TAG(MEDIUM_TAG), wavefield>(field,         \
                                                                deltatover2);  \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      APPLY_CORRECTOR_PHASE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
          MEDIUM_TAG_ELASTIC_PSV_T))

#undef APPLY_CORRECTOR_PHASE

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::forward;

#define APPLY_PREDICTOR_PHASE(DIMENSION_TAG, MEDIUM_TAG)                       \
  if (tag == GET_TAG(MEDIUM_TAG)) {                                            \
    return predictor_phase_impl<GET_TAG(MEDIUM_TAG), wavefield>(               \
        field, deltat, deltatover2, deltasquareover2);                         \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      APPLY_PREDICTOR_PHASE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
          MEDIUM_TAG_ELASTIC_PSV_T))

#undef APPLY_PREDICTOR_PHASE

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::adjoint;

#define APPLY_CORRECTOR_PHASE(DIMENSION_TAG, MEDIUM_TAG)                       \
  if (tag == GET_TAG(MEDIUM_TAG)) {                                            \
    return corrector_phase_impl<GET_TAG(MEDIUM_TAG), wavefield>(adjoint_field, \
                                                                deltatover2);  \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      APPLY_CORRECTOR_PHASE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
          MEDIUM_TAG_ELASTIC_PSV_T))

#undef APPLY_CORRECTOR_PHASE

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_corrector_phase_backward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::backward;

#define APPLY_CORRECTOR_PHASE(DIMENSION_TAG, MEDIUM_TAG)                       \
  if (tag == GET_TAG(MEDIUM_TAG)) {                                            \
    return corrector_phase_impl<GET_TAG(MEDIUM_TAG), wavefield>(               \
        backward_field, -1.0 * deltatover2);                                   \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      APPLY_CORRECTOR_PHASE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
          MEDIUM_TAG_ELASTIC_PSV_T))

#undef APPLY_CORRECTOR_PHASE

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::adjoint;

#define APPLY_PREDICTOR_PHASE(DIMENSION_TAG, MEDIUM_TAG)                       \
  if (tag == GET_TAG(MEDIUM_TAG)) {                                            \
    return predictor_phase_impl<GET_TAG(MEDIUM_TAG), wavefield>(               \
        adjoint_field, deltat, deltatover2, deltasquareover2);                 \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      APPLY_PREDICTOR_PHASE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
          MEDIUM_TAG_ELASTIC_PSV_T))

#undef APPLY_PREDICTOR_PHASE

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_predictor_phase_backward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::backward;

#define APPLY_PREDICTOR_PHASE(DIMENSION_TAG, MEDIUM_TAG)                       \
  if (tag == GET_TAG(MEDIUM_TAG)) {                                            \
    return predictor_phase_impl<GET_TAG(MEDIUM_TAG), wavefield>(               \
        backward_field, -1.0 * deltat, -1.0 * deltatover2, deltasquareover2);  \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      APPLY_PREDICTOR_PHASE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
          MEDIUM_TAG_ELASTIC_PSV_T))

#undef APPLY_PREDICTOR_PHASE

  Kokkos::abort("Medium type not supported.");
  /// Code path should never be reached
  return 0;
}

void specfem::time_scheme::newmark<specfem::simulation::type::forward>::print(
    std::ostream &message) const {
  message << "  Time Scheme:\n"
          << "------------------------------\n"
          << "- Newmark\n"
          << "    simulation type = forward\n"
          << "    dt = " << this->deltat
          << "\n"
          // << "    number of time steps = " << this->nstep << "\n"
          << "    Start time = " << this->t0 << "\n";
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::print(
    std::ostream &message) const {
  message << "  Time Scheme:\n"
          << "------------------------------\n"
          << "- Newmark\n"
          << "    simulation type = adjoint\n"
          << "    dt = " << this->deltat
          << "\n"
          // << "    number of time steps = " << this->nstep << "\n"
          << "    Start time = " << this->t0 << "\n";
}

#endif
