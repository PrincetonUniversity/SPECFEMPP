#include "specfem/injection/fk/solver.hpp"
#include "specfem/injection/fk/impl/driver.hpp"

specfem::injection::fk::FkResult specfem::injection::fk::solve(
    const specfem::injection::fk::LayeredModel &model,
    const specfem::injection::fk::IncidentWave &wave,
    const specfem::injection::fk::TimeWindow &window,
    const specfem::injection::fk::EvalPoints &points, bool compute_traction,
    specfem::injection::fk::field_derivative derivative) {
  model.validate();
  return specfem::injection::fk_impl::run_fk_driver(
      model, wave, window, points, compute_traction, derivative);
}
