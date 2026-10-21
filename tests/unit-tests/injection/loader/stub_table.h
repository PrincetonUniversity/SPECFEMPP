/* Shared, do-nothing ABI functions for the deliberately broken loader test
 * plugins. Each plugin includes this once and only varies the manifest, the
 * table, or the entry point. */

#pragma once

#include "specfem/plugin/injection_abi.h"

static void *stub_create(const char *config_path,
                         const spfp_simulation_info *info,
                         const spfp_communicator *communicator) {
  static int state;
  (void)config_path;
  (void)info;
  (void)communicator;
  return &state;
}

static int32_t stub_supports(void *self, spfp_medium medium,
                             spfp_quantity quantity) {
  (void)self;
  (void)medium;
  (void)quantity;
  return 0;
}

static spfp_status stub_setup(void *self, const spfp_points *points,
                              const spfp_request *request) {
  (void)self;
  (void)points;
  (void)request;
  return spfp_status_ok;
}

static spfp_status stub_fill(void *self, int32_t first_step,
                             const spfp_buffer *buffer) {
  (void)self;
  (void)first_step;
  (void)buffer;
  return spfp_status_ok;
}

static const char *stub_last_error(const void *self) {
  (void)self;
  return "";
}

static void stub_destroy(void *self) { (void)self; }

/* Defines `stub_table` with the given ABI version and name. */
#define STUB_TABLE(ABI_VERSION, NAME)                                          \
  static const spfp_injection_plugin stub_table = { ABI_VERSION,               \
                                                    0u,                        \
                                                    0u,                        \
                                                    NAME,                      \
                                                    "0.0.0",                   \
                                                    stub_create,               \
                                                    stub_supports,             \
                                                    stub_setup,                \
                                                    stub_fill,                 \
                                                    stub_last_error,           \
                                                    stub_destroy }
