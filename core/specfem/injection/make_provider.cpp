#include "specfem/injection/make_provider.hpp"

#include "specfem/injection/fk/time_window.hpp"
#include "specfem/injection/fk_provider.hpp"
#include "specfem/injection/injection_points.hpp"
#include <stdexcept>

template <specfem::element::dimension_tag DimensionTag>
std::shared_ptr<specfem::injection::injection_provider<DimensionTag>>
specfem::injection::make_provider(
    const specfem::injection::injection_inputs &inputs,
    const specfem::injection::injection_points &points, type_real dt,
    int nstep) {
  if (inputs.method == "fk") {
    specfem::injection::fk::TimeWindow window;
    window.dt = dt;
    window.nstep = nstep;
    window.frequency_max = inputs.frequency_max;
    window.frequency_sampling = inputs.frequency_sampling;
    window.time_window_length = inputs.time_window_length;
    return std::make_shared<specfem::injection::fk_provider<DimensionTag>>(
        inputs.model, inputs.wave, window, points, inputs.compute_traction,
        inputs.derivative, inputs.window_steps, inputs.consistency_tolerance);
  }
  throw std::runtime_error("injection: unknown method '" + inputs.method +
                           "'. Supported methods: fk");
}

template std::shared_ptr<specfem::injection::injection_provider<
    specfem::element::dimension_tag::dim2>>
specfem::injection::make_provider<specfem::element::dimension_tag::dim2>(
    const specfem::injection::injection_inputs &,
    const specfem::injection::injection_points &, type_real, int);

template std::shared_ptr<specfem::injection::injection_provider<
    specfem::element::dimension_tag::dim3>>
specfem::injection::make_provider<specfem::element::dimension_tag::dim3>(
    const specfem::injection::injection_inputs &,
    const specfem::injection::injection_points &, type_real, int);
