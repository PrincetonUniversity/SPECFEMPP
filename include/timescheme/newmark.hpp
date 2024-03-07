#ifndef _SPECFEM_TIMESCHEME_NEWMARK_HPP_
#define _SPECFEM_TIMESCHEME_NEWMARK_HPP_

#include "domain/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "specfem_setup.hpp"
#include "timescheme.hpp"
#include <ostream>

namespace specfem {
namespace time_scheme {

template <specfem::simulation::type Simulation> class newmark;

template <>
class newmark<specfem::simulation::type::forward> : public time_scheme {

public:
  using elastic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::elastic>;
  using acoustic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::acoustic>;

  newmark(const int nstep, const int nstep_between_samples, const type_real dt,
          const type_real t0)
      : time_scheme(nstep, nstep_between_samples), deltat(dt),
        deltatover2(dt / 2.0), deltasquareover2(dt * dt / 2.0), t0(t0) {}

  void print(std::ostream &out) const override;

  void apply_predictor_phase_forward(
      const specfem::element::medium_tag tag) override;

  void apply_corrector_phase_forward(
      const specfem::element::medium_tag tag) override;

  void apply_predictor_phase_backward(
      const specfem::element::medium_tag tag) override{};

  void apply_corrector_phase_backward(
      const specfem::element::medium_tag tag) override{};

  void link_assembly(const specfem::compute::assembly &assembly) override {
    elastic = assembly.fields.forward.elastic;
    acoustic = assembly.fields.forward.acoustic;
  }

  specfem::enums::time_scheme::type timescheme() const override {
    return specfem::enums::time_scheme::type::newmark;
  }

  type_real get_timestep() const override { return this->deltat; }

private:
  type_real t0;
  type_real deltat;
  type_real deltatover2;
  type_real deltasquareover2;
  specfem::compute::impl::field_impl<elastic_type> elastic;
  specfem::compute::impl::field_impl<acoustic_type> acoustic;
};

template <>
class newmark<specfem::simulation::type::adjoint> : public time_scheme {

public:
  using elastic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::elastic>;
  using acoustic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::acoustic>;

  newmark(const int nstep, const int nstep_between_samples, const type_real dt,
          const type_real t0)
      : time_scheme(nstep, nstep_between_samples), deltat(dt),
        deltatover2(dt / 2.0), deltasquareover2(dt * dt / 2.0), t0(t0) {}

  void print(std::ostream &out) const override;

  void apply_predictor_phase_forward(
      const specfem::element::medium_tag tag) override;

  void apply_corrector_phase_forward(
      const specfem::element::medium_tag tag) override;

  void apply_predictor_phase_backward(
      const specfem::element::medium_tag tag) override;

  void apply_corrector_phase_backward(
      const specfem::element::medium_tag tag) override;

  void link_assembly(const specfem::compute::assembly &assembly) override {
    adjoint_elastic = assembly.fields.adjoint.elastic;
    adjoint_acoustic = assembly.fields.adjoint.acoustic;
    backward_elastic = assembly.fields.backward.elastic;
    backward_acoustic = assembly.fields.backward.acoustic;
  }

  specfem::enums::time_scheme::type timescheme() const override {
    return specfem::enums::time_scheme::type::newmark;
  }

  type_real get_timestep() const override { return this->deltat; }

private:
  type_real t0;
  type_real deltat;
  type_real deltatover2;
  type_real deltasquareover2;
  specfem::compute::impl::field_impl<elastic_type> adjoint_elastic;
  specfem::compute::impl::field_impl<acoustic_type> adjoint_acoustic;
  specfem::compute::impl::field_impl<elastic_type> backward_elastic;
  specfem::compute::impl::field_impl<acoustic_type> backward_acoustic;
};

} // namespace time_scheme
} // namespace specfem
#endif
