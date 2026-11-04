#include "fk_injection/solver.hpp"
#include "impl/driver.hpp"

fk_injection::FkResult
fk_injection::solve(const fk_injection::LayeredModel &model,
                    const fk_injection::IncidentWave &wave,
                    const fk_injection::TimeWindow &window,
                    const fk_injection::EvalPoints &points,
                    fk_injection::field_derivative derivative) {
  return fk_injection::impl::run_fk_driver(model, wave, window, points,
                                           derivative);
}
