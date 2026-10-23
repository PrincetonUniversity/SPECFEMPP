#pragma once

/*
 * C ABI (version 1) between SPECFEM++ and wavefield-injection plugins.
 *
 * A plugin is a shared library that exports exactly one symbol,
 * specfempp_injection_plugin(), returning a table of plain C function
 * pointers. Everything that crosses the library boundary is a C struct or a
 * flat host array, so a plugin can be built with any compiler and standard
 * library, and may link its own copies of third-party libraries.
 *
 * Lifecycle (every call is collective on all MPI ranks, including ranks that
 * own no injection points):
 *
 *   create -> supports (per requested medium/quantity) -> setup (once)
 *          -> fill (once per window of time steps) -> destroy
 *
 * Conventions:
 *  - SI units; coordinates in the SPECFEM++ mesh frame (z up).
 *  - Step k corresponds to time t0 + k * dt (see spfp_simulation_info).
 *  - spfp_points is only valid during setup; spfp_buffer only during fill.
 *  - fill must produce the same values regardless of window order or size.
 *  - Errors are reported through status codes and last_error; exceptions must
 *    never cross the boundary, and plugins must not call exit, abort,
 *    MPI_Init, or MPI_Finalize.
 *
 * Plugin authors writing C++ should use specfem/plugin/injection_plugin.hpp,
 * which implements this table on top of an abstract base class.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Version of the injection plugin ABI defined by this header. */
#define SPFP_INJECTION_ABI_VERSION 1

/** @brief Version of the @ref spfp_manifest layout. */
#define SPFP_MANIFEST_VERSION 1

/** @brief Manifest @c plugin_kind value for wavefield-injection plugins. */
#define SPFP_PLUGIN_KIND_INJECTION 1

/** @brief Name of the single symbol every injection plugin exports. */
#define SPFP_INJECTION_PLUGIN_ENTRY_NAME "specfempp_injection_plugin"

#if defined(_WIN32)
#define SPFP_EXPORT __declspec(dllexport)
#else
/** @brief Marks the plugin entry point as visible from outside the library. */
#define SPFP_EXPORT __attribute__((visibility("default")))
#endif

/** @brief Status code returned by plugin functions (see spfp_status_values). */
typedef int32_t spfp_status;

/** @brief Values of @c spfp_status. */
enum spfp_status_values {
  spfp_status_ok = 0,         /**< Success */
  spfp_status_error = 1,      /**< Failure; details via last_error */
  spfp_status_unsupported = 2 /**< Request not supported by the plugin */
};

/** @brief Scalar type of a buffer (see spfp_dtype_values). */
typedef int32_t spfp_dtype;

/** @brief Values of @c spfp_dtype. */
enum spfp_dtype_values {
  spfp_dtype_float32 = 0, /**< 32-bit IEEE float */
  spfp_dtype_float64 = 1  /**< 64-bit IEEE float */
};

/** @brief Medium of an injection point (see spfp_medium_values). */
typedef int32_t spfp_medium;

/** @brief Values of @c spfp_medium. */
enum spfp_medium_values {
  spfp_medium_elastic = 0, /**< Elastic solid */
  spfp_medium_acoustic = 1 /**< Acoustic fluid */
};

/** @brief Physical quantity requested from a plugin (see
 * spfp_quantity_values). */
typedef int32_t spfp_quantity;

/** @brief Values of @c spfp_quantity. */
enum spfp_quantity_values {
  spfp_quantity_displacement = 0,  /**< Displacement (3 components) */
  spfp_quantity_velocity = 1,      /**< Velocity (3 components) */
  spfp_quantity_acceleration = 2,  /**< Acceleration (3 components) */
  spfp_quantity_traction = 3,      /**< Traction on the point's normal (3) */
  spfp_quantity_potential_rate = 4 /**< Fluid potential rate (1 component) */
};

/**
 * @brief MPI communicator handed to the plugin.
 *
 * The communicator is passed as its Fortran handle (@c MPI_Comm_c2f) so this
 * header does not depend on @c mpi.h. A plugin that uses MPI converts it back
 * with @c MPI_Comm_f2c and must use the same MPI implementation as SPECFEM++.
 */
typedef struct spfp_communicator {
  int32_t has_mpi;        /**< 0 when SPECFEM++ runs without MPI */
  int32_t rank;           /**< Rank of this process in the communicator */
  int32_t size;           /**< Number of ranks in the communicator */
  int32_t fortran_handle; /**< MPI_Comm_c2f(comm); unused when has_mpi == 0 */
} spfp_communicator;

/** @brief Simulation parameters shared with the plugin at creation. */
typedef struct spfp_simulation_info {
  int32_t dimension; /**< Spatial dimension (2 or 3) */
  int32_t nstep;     /**< Total number of time steps */
  double t0;         /**< Time of step 0 in s */
  double dt;         /**< Time step in s */
  double bounds[6];  /**< Domain box: xmin, xmax, ymin, ymax, zmin, zmax */
} spfp_simulation_info;

/**
 * @brief Points at which the incident field is requested.
 *
 * Host arrays of length @c n, read-only, valid only during @c setup. Normals
 * are outward unit normals of the boundary the point lies on. Material values
 * are the local mesh values, so a plugin can check them against its own
 * model.
 */
typedef struct spfp_points {
  int64_t n;             /**< Number of points */
  const double *x;       /**< x-coordinates in m */
  const double *y;       /**< y-coordinates in m */
  const double *z;       /**< z-coordinates in m (up) */
  const double *nx;      /**< Unit normal, x-component */
  const double *ny;      /**< Unit normal, y-component */
  const double *nz;      /**< Unit normal, z-component */
  const int32_t *medium; /**< Medium of each point (spfp_medium) */
  const double *rho;     /**< Local density in kg/m^3 */
  const double *vp;      /**< Local P-wave speed in m/s */
  const double *vs;      /**< Local S-wave speed in m/s (0 for fluids) */
} spfp_points;

/**
 * @brief Quantities requested per medium.
 *
 * For a point of a given medium, the buffer components are the requested
 * quantities in list order, each expanded to its component count (3 for
 * vector quantities, 1 for @c spfp_quantity_potential_rate).
 */
typedef struct spfp_request {
  int32_t n_elastic;             /**< Number of elastic quantities */
  const spfp_quantity *elastic;  /**< Quantities for elastic points */
  int32_t n_acoustic;            /**< Number of acoustic quantities */
  const spfp_quantity *acoustic; /**< Quantities for acoustic points */
} spfp_request;

/**
 * @brief Host buffer the plugin fills for one window of time steps.
 *
 * Element (point p, component c, step s) of the window lives at
 * @c data[p * stride_point + c * stride_component + s * stride_step], with
 * strides counted in elements of @c dtype. Step s of the window corresponds
 * to simulation step @c first_step + s of the @c fill call.
 */
typedef struct spfp_buffer {
  void *data;               /**< Host memory to write */
  spfp_dtype dtype;         /**< Element type */
  int32_t ncomponents;      /**< Components per point */
  int64_t npoints;          /**< Number of points */
  int32_t nsteps;           /**< Steps in this window */
  int32_t reserved;         /**< Must be 0 */
  int64_t stride_point;     /**< Stride between points */
  int64_t stride_component; /**< Stride between components */
  int64_t stride_step;      /**< Stride between steps */
} spfp_buffer;

/**
 * @brief Function table describing an injection plugin.
 *
 * Returned by the plugin's exported @c specfempp_injection_plugin(). All
 * function pointers are required.
 */
typedef struct spfp_injection_plugin {
  uint32_t abi_version;  /**< Must equal SPFP_INJECTION_ABI_VERSION */
  uint32_t reserved;     /**< Must be 0 */
  uint64_t capabilities; /**< Capability flags; 0 in ABI v1 */
  const char *name;      /**< Plugin name */
  const char *version;   /**< Plugin version string */

  /** Create a plugin instance; returns NULL on failure (see last_error). */
  void *(*create)(const char *config_path, const spfp_simulation_info *info,
                  const spfp_communicator *communicator);

  /** Return 1 if the plugin can provide @p quantity for @p medium, else 0. */
  int32_t (*supports)(void *self, spfp_medium medium, spfp_quantity quantity);

  /** Receive the injection points and the per-medium request (once). */
  spfp_status (*setup)(void *self, const spfp_points *points,
                       const spfp_request *request);

  /** Fill steps [first_step, first_step + buffer->nsteps) into @p buffer. */
  spfp_status (*fill)(void *self, int32_t first_step,
                      const spfp_buffer *buffer);

  /** Last error message of @p self; with NULL, the last failed create. */
  const char *(*last_error)(const void *self);

  /** Destroy an instance returned by create (NULL is a no-op). */
  void (*destroy)(void *self);
} spfp_injection_plugin;

/** @brief Signature of the exported plugin entry point. */
typedef const spfp_injection_plugin *(*spfp_injection_plugin_entry)(void);

/**
 * @brief Manifest embedded in every plugin, readable without loading it.
 *
 * SPECFEM++ scans the plugin file for the magic bytes and checks the ABI
 * version before calling @c dlopen, so a mismatched plugin never runs code in
 * the simulation process.
 */
typedef struct spfp_manifest {
  char magic[16];            /**< SPFP_MANIFEST_MAGIC_INITIALIZER bytes */
  uint32_t manifest_version; /**< SPFP_MANIFEST_VERSION */
  uint32_t abi_version;      /**< SPFP_INJECTION_ABI_VERSION */
  uint32_t plugin_kind;      /**< SPFP_PLUGIN_KIND_INJECTION */
  uint32_t reserved;         /**< Must be 0 */
  char name[64];             /**< NUL-terminated plugin name */
  char version[32];          /**< NUL-terminated plugin version */
} spfp_manifest;

/*
 * The magic is spelled out byte by byte so that it never appears in a plugin
 * as a string literal: the manifest must be the only place a scanner finds it.
 */
#define SPFP_MANIFEST_MAGIC_INITIALIZER                                        \
  { 'S', 'P', 'E', 'C', 'F', 'E', 'M', 'P',                                    \
    'P', '_', 'P', 'L', 'U', 'G', 'I', 'N' }

#if defined(__APPLE__)
#define SPFP_MANIFEST_SECTION "__DATA,__specfempp"
#else
#define SPFP_MANIFEST_SECTION ".specfempp_plugin"
#endif

#if defined(__has_attribute) && !defined(__APPLE__)
#if __has_attribute(retain)
#define SPFP_MANIFEST_RETAIN __attribute__((retain))
#endif
#endif
#ifndef SPFP_MANIFEST_RETAIN
#define SPFP_MANIFEST_RETAIN
#endif

/**
 * @brief Define the plugin manifest (once per plugin, at file scope).
 *
 * @param NAME    String literal, at most 63 characters.
 * @param VERSION String literal, at most 31 characters.
 */
#define SPFP_DEFINE_INJECTION_MANIFEST(NAME, VERSION)                          \
  __attribute__((used, section(SPFP_MANIFEST_SECTION)))                        \
  SPFP_MANIFEST_RETAIN static const spfp_manifest spfp_injection_manifest = {  \
    SPFP_MANIFEST_MAGIC_INITIALIZER,                                           \
    SPFP_MANIFEST_VERSION,                                                     \
    SPFP_INJECTION_ABI_VERSION,                                                \
    SPFP_PLUGIN_KIND_INJECTION,                                                \
    0u,                                                                        \
    NAME,                                                                      \
    VERSION                                                                    \
  }

/** @brief The plugin entry point; every injection plugin defines it. */
SPFP_EXPORT const spfp_injection_plugin *specfempp_injection_plugin(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
