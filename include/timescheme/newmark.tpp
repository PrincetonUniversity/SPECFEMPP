#ifndef _SPECFEM_TIMESCHEME_NEWMARK_TPP_
#define _SPECFEM_TIMESCHEME_NEWMARK_TPP_

#include "timescheme/newmark.hpp"

namespace {
template <specfem::element::medium_tag MediumType,
          specfem::wavefield::type WavefieldType>
void corrector_phase_impl(
    const specfem::compute::simulation_field<WavefieldType> &field,
    const type_real deltatover2) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;
  const int nglob = field.template get_nglob<MediumType>();
  using LoadFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumType, false,
                            false, true, false>;
  using AddFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumType, false,
                            true, false, false>;

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::corrector_phase_impl",
      specfem::kokkos::DeviceRange(0, nglob), KOKKOS_LAMBDA(const int iglob) {
        LoadFieldType load;
        AddFieldType add;

        specfem::compute::load_on_device(iglob, field, load);

        for (int idim = 0; idim < components; ++idim) {
          add.velocity(idim) += deltatover2 * load.acceleration(idim);
        }

        specfem::compute::add_on_device(iglob, add, field);
      });

  return;
}

template <specfem::element::medium_tag MediumType,
          specfem::wavefield::type WavefieldType>
void predictor_phase_impl(
    const specfem::compute::simulation_field<WavefieldType> &field,
    const type_real deltat, const type_real deltatover2,
    const type_real deltasquareover2) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;
  const int nglob = field.template get_nglob<MediumType>();
  using LoadFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumType, false,
                            true, true, false>;
  using AddFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumType, true,
                            true, false, false>;
  using StoreFieldType =
      specfem::point::field<specfem::dimension::type::dim2, MediumType, false,
                            false, true, false>;

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::predictor_phase_impl",
      specfem::kokkos::DeviceRange(0, nglob), KOKKOS_LAMBDA(const int iglob) {
        LoadFieldType load;
        AddFieldType add;
        StoreFieldType store;

        specfem::compute::load_on_device(iglob, field, load);

        for (int idim = 0; idim < components; ++idim) {
          add.displacement(idim) += deltat * load.velocity(idim) +
                                    deltasquareover2 * load.acceleration(idim);

          add.velocity(idim) += deltatover2 * load.acceleration(idim);

          store.acceleration(idim) = 0;
        }

        specfem::compute::add_on_device(iglob, add, field);
        specfem::compute::store_on_device(iglob, store, field);
      });

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

  constexpr auto wavefield = specfem::wavefield::type::forward;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic) {
    corrector_phase_impl<elastic, wavefield>(field, deltatover2);
  } else if (tag == acoustic) {
    corrector_phase_impl<acoustic, wavefield>(field, deltatover2);
  } else {
    static_assert("medium type not supported");
  }

  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto wavefield = specfem::wavefield::type::forward;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic) {
    predictor_phase_impl<elastic, wavefield>(field, deltat, deltatover2,
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
  constexpr auto wavefield = specfem::wavefield::type::adjoint;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic) {
    corrector_phase_impl<elastic, wavefield>(adjoint_field, deltatover2);
  } else if (tag == acoustic) {
    corrector_phase_impl<acoustic, wavefield>(adjoint_field, deltatover2);
  } else {
    static_assert("medium type not supported");
  }

  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_corrector_phase_backward(const specfem::element::medium_tag tag) {
  constexpr auto wavefield = specfem::wavefield::type::backward;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic) {
    corrector_phase_impl<elastic, wavefield>(backward_field,
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

  constexpr auto wavefield = specfem::wavefield::type::adjoint;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic) {
    predictor_phase_impl<elastic, wavefield>(adjoint_field, deltat, deltatover2,
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
  constexpr auto wavefield = specfem::wavefield::type::backward;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;

  if (tag == elastic) {
    predictor_phase_impl<elastic, wavefield>(
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
