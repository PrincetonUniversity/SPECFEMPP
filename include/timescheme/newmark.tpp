#ifndef _SPECFEM_TIMESCHEME_NEWMARK_TPP_
#define _SPECFEM_TIMESCHEME_NEWMARK_TPP_

#include "timescheme/newmark.hpp"

void specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_corrector_phase(const specfem::element::medium_tag tag) {

  const auto field_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return elastic.field_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return acoustic.field_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(elastic.field_dot){};
  }();

  const auto field_dot_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return elastic.field_dot_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return acoustic.field_dot_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(elastic.field_dot_dot){};
  }();

  const int nglob = field_dot.extent(0);
  const int components = field_dot.extent(1);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::apply_corrector_phase",
      specfem::kokkos::DeviceRange(0, components * nglob),
      KOKKOS_CLASS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        field_dot(iglob, idim) += deltatover2 * field_dot_dot(iglob, idim);
      });

  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_predictor_phase(const specfem::element::medium_tag tag) {
  const auto field = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return elastic.field;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return acoustic.field;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(elastic.field){};
  }();

  const auto field_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return elastic.field_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return acoustic.field_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(elastic.field_dot){};
  }();

  const auto field_dot_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return elastic.field_dot_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return acoustic.field_dot_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(elastic.field_dot_dot){};
  }();

  const int nglob = field.extent(0);
  const int components = field.extent(1);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::apply_predictor_phase",
      specfem::kokkos::DeviceRange(0, components * nglob),
      KOKKOS_CLASS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        field(iglob, idim) += deltat * field_dot(iglob, idim) +
                              deltasquareover2 * field_dot_dot(iglob, idim);

        field_dot(iglob, idim) += deltatover2 * field_dot_dot(iglob, idim);

        field_dot_dot(iglob, idim) = 0;
      });
  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::forward>::print(
    std::ostream &message) const {
  message << "  Time Scheme:\n"
          << "------------------------------\n"
          << "- Newmark\n"
          << "    simulation type = forward\n"
          << "    dt = " << this->deltat << "\n"
          // << "    number of time steps = " << this->nstep << "\n"
          << "    Start time = " << this->t0 << "\n";
}

#endif
