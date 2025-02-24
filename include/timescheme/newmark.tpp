#ifndef _SPECFEM_TIMESCHEME_NEWMARK_TPP_
#define _SPECFEM_TIMESCHEME_NEWMARK_TPP_

#include "parallel_configuration/range_config.hpp"
#include "policies/range.hpp"
#include "timescheme/newmark.hpp"

namespace {
template <specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
void corrector_phase_impl(
    const specfem::compute::simulation_field<WavefieldType> &field,
    const type_real deltatover2) {

  constexpr int components =
      specfem::element::attributes<specfem::dimension::type::dim2, MediumTag>::components();
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

  // Kokkos::parallel_for(
  //     "specfem::TimeScheme::Newmark::corrector_phase_impl",
  //     specfem::kokkos::DeviceRange(0, nglob), KOKKOS_LAMBDA(const int iglob)
  //     {
  //       LoadFieldType load;
  //       AddFieldType add;

  //       specfem::compute::load_on_device(iglob, field, load);

  //       for (int idim = 0; idim < components; ++idim) {
  //         add.velocity(idim) += deltatover2 * load.acceleration(idim);
  //       }

  //       specfem::compute::add_on_device(iglob, add, field);
  //     });

  return;
}

template <specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
void predictor_phase_impl(
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

  // Kokkos::parallel_for(
  //     "specfem::TimeScheme::Newmark::predictor_phase_impl",
  //     specfem::kokkos::DeviceRange(0, nglob), KOKKOS_LAMBDA(const int iglob)
  //     {
  //       LoadFieldType load;
  //       AddFieldType add;
  //       StoreFieldType store;

  //       specfem::compute::load_on_device(iglob, field, load);

  //       for (int idim = 0; idim < components; ++idim) {
  //         add.displacement(idim) += deltat * load.velocity(idim) +
  //                                   deltasquareover2 *
  //                                   load.acceleration(idim);

  //         add.velocity(idim) += deltatover2 * load.acceleration(idim);

  //         store.acceleration(idim) = 0;
  //       }

  //       specfem::compute::add_on_device(iglob, add, field);
  //       specfem::compute::store_on_device(iglob, store, field);
  //     });

  return;
}

// void corrector_phase_impl(
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
//     field_dot_dot, const type_real deltatover2) {
//   const int nglob = field_dot.extent(0);
//   const int components = field_dot.extent(1);

//   Kokkos::parallel_for(
//       "specfem::TimeScheme::Newmark::corrector_phase_impl",
//       specfem::kokkos::DeviceRange(0, components * nglob),
//       KOKKOS_LAMBDA(const int in) {
//         const int iglob = in % nglob;
//         const int idim = in / nglob;
//         field_dot(iglob, idim) += deltatover2 * field_dot_dot(iglob, idim);
//       });
// }

// void predictor_phase_impl(
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
//     specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
//     field_dot_dot, const type_real deltat, const type_real deltatover2, const
//     type_real deltasquareover2) {
//   const int nglob = field.extent(0);
//   const int components = field.extent(1);

//   Kokkos::parallel_for(
//       "specfem::TimeScheme::Newmark::predictor_phase_impl",
//       specfem::kokkos::DeviceRange(0, components * nglob),
//       KOKKOS_LAMBDA(const int in) {
//         const int iglob = in % nglob;
//         const int idim = in / nglob;
//         field(iglob, idim) += deltat * field_dot(iglob, idim) +
//                               deltasquareover2 * field_dot_dot(iglob, idim);

//         field_dot(iglob, idim) += deltatover2 * field_dot_dot(iglob, idim);

//         field_dot_dot(iglob, idim) = 0;
//       });
// }
} // namespace

void specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::forward;
  constexpr auto elastic_sv = specfem::element::medium_tag::elastic_sv;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic_sv) {
    corrector_phase_impl<elastic_sv, wavefield>(field, deltatover2);
  } else if (tag == acoustic) {
    corrector_phase_impl<acoustic, wavefield>(field, deltatover2);
  } else {
    static_assert("medium type not supported");
  }

  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::forward;
  constexpr auto elastic_sv = specfem::element::medium_tag::elastic_sv;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic_sv) {
    predictor_phase_impl<elastic_sv, wavefield>(field, deltat, deltatover2,
                                             deltasquareover2);
  } else if (tag == acoustic) {
    predictor_phase_impl<acoustic, wavefield>(field, deltat, deltatover2,
                                              deltasquareover2);
  } else {
    static_assert("medium type not supported");
  }
  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {
  constexpr auto wavefield = specfem::wavefield::simulation_field::adjoint;
  constexpr auto elastic_sv = specfem::element::medium_tag::elastic_sv;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic_sv) {
    corrector_phase_impl<elastic_sv, wavefield>(adjoint_field, deltatover2);
  } else if (tag == elastic_sv) {
    corrector_phase_impl<acoustic, wavefield>(adjoint_field, deltatover2);
  } else {
    static_assert("medium type not supported");
  }

  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_corrector_phase_backward(const specfem::element::medium_tag tag) {
  constexpr auto wavefield = specfem::wavefield::simulation_field::backward;
  constexpr auto elastic_sv = specfem::element::medium_tag::elastic_sv;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic_sv) {
    corrector_phase_impl<elastic_sv, wavefield>(backward_field,
                                             -1.0 * deltatover2);
  } else if (tag == acoustic) {
    corrector_phase_impl<acoustic, wavefield>(backward_field,
                                              -1.0 * deltatover2);
  } else {
    static_assert("medium type not supported");
  }

  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::simulation_field::adjoint;
  constexpr auto elastic_sv = specfem::element::medium_tag::elastic_sv;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic_sv) {
    predictor_phase_impl<elastic_sv, wavefield>(adjoint_field, deltat, deltatover2,
                                             deltasquareover2);
  } else if (tag == acoustic) {
    predictor_phase_impl<acoustic, wavefield>(adjoint_field, deltat,
                                              deltatover2, deltasquareover2);
  } else {
    static_assert("medium type not supported");
  }
  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_predictor_phase_backward(const specfem::element::medium_tag tag) {
  constexpr auto wavefield = specfem::wavefield::simulation_field::backward;
  constexpr auto elastic_sv = specfem::element::medium_tag::elastic_sv;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic_sv) {
    predictor_phase_impl<elastic_sv, wavefield>(
        backward_field, -1.0 * deltat, -1.0 * deltatover2, deltasquareover2);
  } else if (tag == acoustic) {
    predictor_phase_impl<acoustic, wavefield>(
        backward_field, -1.0 * deltat, -1.0 * deltatover2, deltasquareover2);
  } else {
    static_assert("medium type not supported");
  }
  return;
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
