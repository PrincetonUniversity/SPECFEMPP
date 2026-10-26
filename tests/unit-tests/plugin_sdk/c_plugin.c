/* Test plugin written against the raw C ABI, without the C++ layer. It
 * provides elastic isotropic displacement only, with value 42 + step, and
 * reads the layout SPECFEM++ sends instead of assuming one. */

#include "specfem/plugin/injection_abi.h"
#include <stdlib.h>
#include <string.h>

typedef struct c_plugin_state {
  char last_error[128];
} c_plugin_state;

static void *c_plugin_create(const char *config_path,
                             const specfempp_simulation_info *info,
                             const specfempp_communicator *communicator) {
  (void)config_path;
  (void)communicator;
  if (info->dimension != 3)
    return NULL;
  return calloc(1, sizeof(c_plugin_state));
}

static int32_t c_plugin_supports(void *self, specfempp_medium medium,
                                 specfempp_property property,
                                 specfempp_quantity quantity) {
  (void)self;
  return medium == specfempp_medium_elastic &&
         property == specfempp_property_isotropic &&
         quantity == specfempp_quantity_displacement;
}

static specfempp_status c_plugin_setup(void *self,
                                       const specfempp_points *points) {
  c_plugin_state *state = (c_plugin_state *)self;
  for (int32_t g = 0; g < points->ngroups; ++g) {
    const specfempp_point_group *group = points->groups[g];
    if (group->medium != specfempp_medium_elastic ||
        group->property != specfempp_property_isotropic ||
        group->nslots != 1 ||
        group->slots[0].quantity != specfempp_quantity_displacement) {
      strcpy(state->last_error, "only elastic isotropic displacement");
      return specfempp_status_unsupported;
    }
  }
  state->last_error[0] = '\0';
  return specfempp_status_ok;
}

static specfempp_status
c_plugin_fill(void *self, int32_t first_step, int32_t nbuffers,
              const specfempp_buffer *const *buffers) {
  c_plugin_state *state = (c_plugin_state *)self;
  for (int32_t g = 0; g < nbuffers; ++g) {
    const specfempp_buffer *buffer = buffers[g];
    double *data = (double *)buffer->data;
    if (buffer->dtype != specfempp_dtype_float64) {
      strcpy(state->last_error, "only float64 buffers");
      return specfempp_status_unsupported;
    }
    for (int64_t point = 0; point < buffer->npoints; ++point)
      for (int32_t component = 0; component < buffer->ncomponents;
           ++component)
        for (int32_t step = 0; step < buffer->nsteps; ++step)
          data[point * buffer->stride_point +
               component * buffer->stride_component +
               step * buffer->stride_step] = 42.0 + first_step + step;
  }
  return specfempp_status_ok;
}

static const char *c_plugin_last_error(const void *self) {
  return self == NULL ? "" : ((const c_plugin_state *)self)->last_error;
}

static void c_plugin_destroy(void *self) { free(self); }

SPECFEMPP_DEFINE_INJECTION_MANIFEST("test-c", "0.1.0");

static const specfempp_injection_plugin_table c_plugin_table = {
  sizeof(specfempp_injection_plugin_table),
  SPECFEMPP_INJECTION_ABI_VERSION,
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

SPECFEMPP_EXPORT const specfempp_injection_plugin_table *
specfempp_injection_plugin(void) {
  return &c_plugin_table;
}
