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
  std::vector<std::int64_t> group_begin;
  std::string error;
};

std::string &mock_create_error() {
  static std::string error;
  return error;
}

void *mock_create(const char *config_path, const specfem_simulation_info *info,
                  const specfem_communicator *communicator) {
  ++mock_calls().create;
  mock_calls().info = *info;
  mock_calls().communicator = *communicator;
  if (std::string(config_path) == "fail-create") {
    mock_create_error() = "mock create failed";
    return nullptr;
  }
  return new MockState{ config_path, communicator->rank, {}, {}, {} };
}

std::int32_t mock_supports(void *self, specfem_medium medium,
                           [[maybe_unused]] specfem_property property,
                           specfem_quantity quantity) {
  const auto *state = static_cast<MockState *>(self);
  if (state->config == "no-traction")
    return quantity != specfem_quantity_traction &&
           quantity != specfem_quantity_potential_rate;
  return (medium == specfem_medium_acoustic) ==
         (quantity == specfem_quantity_potential_rate);
}

specfem_status mock_setup(void *self, const specfem_points *points) {
  auto *state = static_cast<MockState *>(self);
  ++mock_calls().setup;
  mock_calls().setup_points = points->n;
  mock_calls().groups.clear();
  for (std::int32_t g = 0; g < points->ngroups; ++g) {
    const auto &group = *points->groups[g];
    MockGroup copy{ group.medium,
                    group.property,
                    group.begin,
                    group.count,
                    group.ncomponents,
                    {},
                    {},
                    {} };
    for (std::int32_t k = 0; k < group.nproperties; ++k)
      copy.property_names.emplace_back(group.property_names[k]);
    if (group.properties != nullptr)
      copy.properties.assign(
          group.properties, group.properties + group.count * group.nproperties);
    copy.slots.assign(group.slots, group.slots + group.nslots);
    mock_calls().groups.push_back(copy);
    state->group_begin.push_back(group.begin);
  }
  if (state->config == "fail-setup") {
    state->error = "mock setup failed";
    return specfem_status_error;
  }
  state->x.assign(points->x, points->x + points->n);
  return specfem_status_ok;
}

specfem_status mock_fill(void *self, std::int32_t first_step,
                         std::int32_t nbuffers,
                         const specfem_buffer *const *buffers) {
  auto *state = static_cast<MockState *>(self);
  ++mock_calls().fill;
  mock_calls().fill_starts.push_back(first_step);
  if (state->config == "fail-fill" ||
      (state->config == "fail-fill-rank0" && state->rank == 0)) {
    state->error = "mock fill failed";
    return specfem_status_error;
  }
  for (std::int32_t g = 0; g < nbuffers; ++g) {
    const auto &buffer = *buffers[g];
    for (std::int64_t point = 0; point < buffer.npoints; ++point) {
      const std::int64_t row = state->group_begin[g] + point;
      for (std::int32_t slot = 0; slot < buffer.ncomponents; ++slot)
        for (std::int32_t step = 0; step < buffer.nsteps; ++step) {
          double value = mock_value(state->x[row], slot, first_step + step);
          if (state->config == "nan-fill" && row == 1 && slot == 2 &&
              first_step + step == 5)
            value = std::numeric_limits<double>::quiet_NaN();
          const std::int64_t offset = point * buffer.stride_point +
                                      slot * buffer.stride_component +
                                      step * buffer.stride_step;
          if (buffer.dtype == specfem_dtype_float32)
            static_cast<float *>(buffer.data)[offset] =
                static_cast<float>(value);
          else
            static_cast<double *>(buffer.data)[offset] = value;
        }
    }
  }
  return specfem_status_ok;
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
  static const specfem_injection_plugin_table table = {
    .struct_size = sizeof(specfem_injection_plugin_table),
    .abi_version = SPECFEM_INJECTION_ABI_VERSION,
    .capabilities = 0,
    .name = "session-mock",
    .version = "1.0.0",
    .create = mock_create,
    .supports = mock_supports,
    .setup = mock_setup,
    .fill = mock_fill,
    .last_error = mock_last_error,
    .destroy = mock_destroy
  };
  [[maybe_unused]] static const bool registered = [] {
    specfem::injection::register_static_plugin(&table);
    return true;
  }();
  return "static:session-mock";
}

} // namespace specfem::injection_session_test
