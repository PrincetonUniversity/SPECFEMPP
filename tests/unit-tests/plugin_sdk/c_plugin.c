/* Test plugin written against the raw C ABI, without the C++ layer. It
 * provides elastic displacement only, with value 42 + step. */

#include "specfem/plugin/injection_abi.h"
#include <stdlib.h>

typedef struct c_plugin_state {
  int64_t npoints;
} c_plugin_state;

static void *c_plugin_create(const char *config_path,
                             const spfp_simulation_info *info,
                             const spfp_communicator *communicator) {
  (void)config_path;
  (void)info;
  (void)communicator;
  return calloc(1, sizeof(c_plugin_state));
}

static int32_t c_plugin_supports(void *self, spfp_medium medium,
                                 spfp_quantity quantity) {
  (void)self;
  return medium == spfp_medium_elastic &&
         quantity == spfp_quantity_displacement;
}

static spfp_status c_plugin_setup(void *self, const spfp_points *points,
                                  const spfp_request *request) {
  if (request->n_elastic != 1 ||
      request->elastic[0] != spfp_quantity_displacement)
    return spfp_status_unsupported;
  ((c_plugin_state *)self)->npoints = points->n;
  return spfp_status_ok;
}

static spfp_status c_plugin_fill(void *self, int32_t first_step,
                                 const spfp_buffer *buffer) {
  const c_plugin_state *state = (const c_plugin_state *)self;
  double *data = (double *)buffer->data;
  if (buffer->dtype != spfp_dtype_float64)
    return spfp_status_unsupported;
  for (int64_t point = 0; point < state->npoints; ++point)
    for (int32_t component = 0; component < 3; ++component)
      for (int32_t step = 0; step < buffer->nsteps; ++step)
        data[point * buffer->stride_point +
             component * buffer->stride_component +
             step * buffer->stride_step] = 42.0 + first_step + step;
  return spfp_status_ok;
}

static const char *c_plugin_last_error(const void *self) {
  (void)self;
  return "";
}

static void c_plugin_destroy(void *self) { free(self); }

SPFP_DEFINE_INJECTION_MANIFEST("test-c", "0.1.0");

static const spfp_injection_plugin c_plugin_table = {
  SPFP_INJECTION_ABI_VERSION,
  0u,
  0u,
  "test-c",
  "0.1.0",
  c_plugin_create,
  c_plugin_supports,
  c_plugin_setup,
  c_plugin_fill,
  c_plugin_last_error,
  c_plugin_destroy
};

SPFP_EXPORT const spfp_injection_plugin *specfempp_injection_plugin(void) {
  return &c_plugin_table;
}
