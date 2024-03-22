#ifndef _SPECFEM_TIMESCHEME_NEWMARK_TPP_
#define _SPECFEM_TIMESCHEME_NEWMARK_TPP_

#include "timescheme/newmark.hpp"

namespace {
void corrector_phase_impl(
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    const type_real deltatover2) {
  const int nglob = field_dot.extent(0);
  const int components = field_dot.extent(1);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::corrector_phase_impl",
      specfem::kokkos::DeviceRange(0, components * nglob),
      KOKKOS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        field_dot(iglob, idim) += deltatover2 * field_dot_dot(iglob, idim);
      });
}

void predictor_phase_impl(
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    const type_real deltat, const type_real deltatover2,
    const type_real deltasquareover2) {
  const int nglob = field.extent(0);
  const int components = field.extent(1);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::predictor_phase_impl",
      specfem::kokkos::DeviceRange(0, components * nglob),
      KOKKOS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        field(iglob, idim) += deltat * field_dot(iglob, idim) +
                              deltasquareover2 * field_dot_dot(iglob, idim);

        field_dot(iglob, idim) += deltatover2 * field_dot_dot(iglob, idim);

        field_dot_dot(iglob, idim) = 0;
      });
}
} // namespace

void specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

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

  corrector_phase_impl(field_dot, field_dot_dot, deltatover2);

  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::forward>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {
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

  predictor_phase_impl(field, field_dot, field_dot_dot, deltat, deltatover2,
                       deltasquareover2);
  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {
  const auto field_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return adjoint_elastic.field_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return adjoint_acoustic.field_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(adjoint_elastic.field_dot){};
  }();

  const auto field_dot_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return adjoint_elastic.field_dot_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return adjoint_acoustic.field_dot_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(adjoint_elastic.field_dot_dot){};
  }();

  corrector_phase_impl(field_dot, field_dot_dot, deltatover2);

  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_corrector_phase_backward(const specfem::element::medium_tag tag) {
  const auto field_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return backward_elastic.field_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return backward_acoustic.field_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(backward_elastic.field_dot){};
  }();

  const auto field_dot_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return backward_elastic.field_dot_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return backward_acoustic.field_dot_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(backward_elastic.field_dot_dot){};
  }();

  corrector_phase_impl(field_dot, field_dot_dot, -1.0 * deltatover2);

  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {
  const auto field = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return adjoint_elastic.field;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return adjoint_acoustic.field;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(adjoint_elastic.field){};
  }();

  const auto field_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return adjoint_elastic.field_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return adjoint_acoustic.field_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(adjoint_elastic.field_dot){};
  }();

  const auto field_dot_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return adjoint_elastic.field_dot_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return adjoint_acoustic.field_dot_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(adjoint_elastic.field_dot_dot){};
  }();

  predictor_phase_impl(field, field_dot, field_dot_dot, deltat, deltatover2,
                       deltasquareover2);
  return;
}

void specfem::time_scheme::newmark<specfem::simulation::type::combined>::
    apply_predictor_phase_backward(const specfem::element::medium_tag tag) {
  const auto field = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return backward_elastic.field;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return backward_acoustic.field;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(backward_elastic.field){};
  }();

  const auto field_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return backward_elastic.field_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return backward_acoustic.field_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(backward_elastic.field_dot){};
  }();

  const auto field_dot_dot = [&]() -> auto {
    if (tag == specfem::element::medium_tag::elastic) {
      return backward_elastic.field_dot_dot;
    } else if (tag == specfem::element::medium_tag::acoustic) {
      return backward_acoustic.field_dot_dot;
    } else {
      static_assert("medium type not supported");
    }
    return decltype(backward_elastic.field_dot_dot){};
  }();

  predictor_phase_impl(field, field_dot, field_dot_dot, -1.0 * deltat,
                       -1.0 * deltatover2, deltasquareover2);
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
