#include "mock_plugin.hpp"
#include "specfem/injection.hpp"
#include <cmath>
#include <limits>
#include <string>

namespace specfem::injection_session_test {

// Per-instance state of the mock plugin.
struct MockState {
  std::string config;
  int rank = 0;
  std::vector<double> x;
  std::string error;
};

std::string &mock_create_error() {
  static std::string error;
  return error;
}

void *mock_create(const char *config_path, const spfp_simulation_info *info,
                  const spfp_communicator *communicator) {
  ++mock_calls().create;
  mock_calls().info = *info;
  mock_calls().communicator = *communicator;
  if (std::string(config_path) == "fail-create") {
    mock_create_error() = "mock create failed";
    return nullptr;
  }
  return new MockState{ config_path, communicator->rank, {}, {} };
}

std::int32_t mock_supports(void *self, spfp_medium medium,
                           spfp_quantity quantity) {
  const auto *state = static_cast<MockState *>(self);
  if (state->config == "no-traction")
    return quantity != spfp_quantity_traction &&
           quantity != spfp_quantity_potential_rate;
  return (medium == spfp_medium_acoustic) ==
         (quantity == spfp_quantity_potential_rate);
}

spfp_status mock_setup(void *self, const spfp_points *points,
                       [[maybe_unused]] const spfp_request *request) {
  auto *state = static_cast<MockState *>(self);
  ++mock_calls().setup;
  mock_calls().setup_points = points->n;
  if (state->config == "fail-setup") {
    state->error = "mock setup failed";
    return spfp_status_error;
  }
  state->x.assign(points->x, points->x + points->n);
  return spfp_status_ok;
}

spfp_status mock_fill(void *self, std::int32_t first_step,
                      const spfp_buffer *buffer) {
  auto *state = static_cast<MockState *>(self);
  ++mock_calls().fill;
  mock_calls().fill_starts.push_back(first_step);
  if (state->config == "fail-fill" ||
      (state->config == "fail-fill-rank0" && state->rank == 0)) {
    state->error = "mock fill failed";
    return spfp_status_error;
  }
  for (std::int64_t point = 0; point < buffer->npoints; ++point)
    for (std::int32_t slot = 0; slot < buffer->ncomponents; ++slot)
      for (std::int32_t step = 0; step < buffer->nsteps; ++step) {
        double value = mock_value(state->x[point], slot, first_step + step);
        if (state->config == "nan-fill" && point == 1 && slot == 2 &&
            first_step + step == 5)
          value = std::numeric_limits<double>::quiet_NaN();
        const std::int64_t offset = point * buffer->stride_point +
                                    slot * buffer->stride_component +
                                    step * buffer->stride_step;
        if (buffer->dtype == spfp_dtype_float32)
          static_cast<float *>(buffer->data)[offset] =
              static_cast<float>(value);
        else
          static_cast<double *>(buffer->data)[offset] = value;
      }
  return spfp_status_ok;
}

const char *mock_last_error(const void *self) {
  if (self == nullptr)
    return mock_create_error().c_str();
  return static_cast<const MockState *>(self)->error.c_str();
}

void mock_destroy(void *self) {
  ++mock_calls().destroy;
  delete static_cast<MockState *>(self);
}

MockCalls &mock_calls() {
  static MockCalls calls;
  return calls;
}

void reset_mock_calls() { mock_calls() = MockCalls{}; }

double mock_value(const double x, const int slot, const int step) {
  return x + 10.0 * slot + 1000.0 * step;
}

const char *mock_plugin_spec() {
  static const spfp_injection_plugin table = { .abi_version =
                                                   SPFP_INJECTION_ABI_VERSION,
                                               .name = "session-mock",
                                               .version = "1.0.0",
                                               .create = mock_create,
                                               .supports = mock_supports,
                                               .setup = mock_setup,
                                               .fill = mock_fill,
                                               .last_error = mock_last_error,
                                               .destroy = mock_destroy };
  [[maybe_unused]] static const bool registered = [] {
    specfem::injection::register_static_plugin(&table);
    return true;
  }();
  return "static:session-mock";
}

} // namespace specfem::injection_session_test
