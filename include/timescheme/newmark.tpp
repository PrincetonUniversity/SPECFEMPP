#pragma once

#include "execution/for_all.hpp"
#include "execution/range_iterator.hpp"
#include "parallel_configuration/range_config.hpp"
#include "timescheme/newmark.hpp"

namespace {
template <specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
int corrector_phase_impl(
    const specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                              WavefieldType> &field,
    const type_real deltatover2) {

  constexpr int components =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   MediumTag>::components;
  const int nglob = field.template get_nglob<MediumTag>();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif
  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2, MediumTag,
                                   using_simd>;

  using PointVelocityType =
      specfem::point::velocity<specfem::dimension::type::dim2, MediumTag,
                               using_simd>;

  using parallel_config = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  specfem::execution::RangeIterator range(parallel_config(), nglob);

  using IndexType = specfem::point::assembly_index<using_simd>;

  Kokkos::Profiling::pushRegion("Compute Corrector Phase");

  specfem::execution::for_all(
      "specfem::TimeScheme::Newmark::corrector_phase_impl", range,
      KOKKOS_LAMBDA(const IndexType &index) {
        PointAccelerationType acceleration;
        PointVelocityType velocity;

        specfem::assembly::load_on_device(index, field, velocity, acceleration);

        for (int idim = 0; idim < components; ++idim) {
          velocity(idim) += deltatover2 * acceleration(idim);
        }

        specfem::assembly::store_on_device(index, field, velocity);
      });

  Kokkos::Profiling::popRegion();

  return nglob * specfem::element::attributes<specfem::dimension::type::dim2,
                                              MediumTag>::components;
}

template <specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
int predictor_phase_impl(
    const specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                              WavefieldType> &field,
    const type_real deltat, const type_real deltatover2,
    const type_real deltasquareover2) {

  constexpr int components =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   MediumTag>::components;
  const int nglob = field.template get_nglob<MediumTag>();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif

  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2, MediumTag,
                                   using_simd>;
  using PointVelocityType =
      specfem::point::velocity<specfem::dimension::type::dim2, MediumTag,
                               using_simd>;
  using PointDisplacementType =
      specfem::point::displacement<specfem::dimension::type::dim2, MediumTag,
                                   using_simd>;

  using parallel_config = specfem::parallel_config::default_range_config<
      specfem::datatype::simd<type_real, using_simd>,
      Kokkos::DefaultExecutionSpace>;

  specfem::execution::RangeIterator range(parallel_config(), nglob);

  using IndexType = specfem::point::assembly_index<using_simd>;

  Kokkos::Profiling::pushRegion("Compute Predictor Phase");

  specfem::execution::for_all(
      "specfem::TimeScheme::Newmark::corrector_phase_impl", range,
      KOKKOS_LAMBDA(const IndexType &index) {
        PointDisplacementType displacement;
        PointVelocityType velocity;
        PointAccelerationType acceleration;

        specfem::assembly::load_on_device(index, field, displacement, velocity,
                                          acceleration);

        for (int idim = 0; idim < components; ++idim) {
          displacement(idim) +=
              deltat * velocity(idim) + deltasquareover2 * acceleration(idim);

          velocity(idim) += deltatover2 * acceleration(idim);
          acceleration(idim) = 0;
        }

        specfem::assembly::store_on_device(index, field, displacement, velocity,
                                           acceleration);
      });

  Kokkos::Profiling::popRegion();

  return nglob * specfem::element::attributes<specfem::dimension::type::dim2,
                                              MediumTag>::components;
}
} // namespace

int specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::forward;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return corrector_phase_impl<_medium_tag_, wavefield>(field,
                                                               deltatover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::forward;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return predictor_phase_impl<_medium_tag_, wavefield>(
              field, deltat, deltatover2, deltasquareover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::adjoint;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return corrector_phase_impl<_medium_tag_, wavefield>(adjoint_field,
                                                               deltatover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_corrector_phase_backward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::backward;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       ELASTIC_PSV_T, POROELASTIC)),
      {
        if (tag == _medium_tag_) {
          return corrector_phase_impl<_medium_tag_, wavefield>(
              backward_field, -1.0 * deltatover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::adjoint;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return predictor_phase_impl<_medium_tag_, wavefield>(
              adjoint_field, deltat, deltatover2, deltasquareover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

int specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_predictor_phase_backward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::backward;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return predictor_phase_impl<_medium_tag_, wavefield>(
              backward_field, -1.0 * deltat, -1.0 * deltatover2,
              deltasquareover2);
        }
      })

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
