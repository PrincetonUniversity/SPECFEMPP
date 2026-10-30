#pragma once

/*
 * C ABI (version 1) between SPECFEM++ and wavefield-injection plugins.
 *
 * A plugin is a shared library that exports exactly one symbol,
 * specfem_injection_plugin(), returning a table of plain C function
 * pointers. Everything that crosses the library boundary is a C struct or a
 * flat host array, so a plugin can be built with any compiler and standard
 * library, and may link its own copies of third-party libraries.
 *
 * Lifecycle (every call is collective on all MPI ranks, including ranks that
 * own no injection points):
 *
 *   create -> supports (per medium, property and quantity) -> setup (once)
 *          -> fill (once per window of time steps) -> destroy
 *
 * Conventions:
 *  - ABI v1 is three-dimensional only (specfem_simulation_info::dimension
 *    is 3).
 *  - Units: every value (coordinates, bounds, times, material properties,
 *    injected fields) is in the units of the SPECFEM++ mesh and model.
 *    SPECFEM++ performs no unit conversion; configure the plugin to match.
 *  - Coordinates are in the SPECFEM++ mesh frame (z up).
 *  - Step k corresponds to time t0 + k * dt (see specfem_simulation_info).
 *  - Points are grouped by material system, i.e. by (medium, property) pair,
 *    mirroring the medium x property framework of SPECFEM++. Every rank sees
 *    the same list of groups; groups without local points have count 0.
 *  - specfem_points is only valid during setup; the buffers only during
 *    fill.
 *  - fill must produce the same values regardless of window order or size.
 *  - Errors are reported through status codes and last_error; exceptions must
 *    never cross the boundary, and plugins must not call exit, abort,
 *    MPI_Init, or MPI_Finalize.
 *
 * Extensibility: every struct passed by pointer starts with a struct_size
 * field that its writer sets to sizeof(struct). Fields are only ever appended,
 * so a reader built against an older header ignores the tail, and a reader
 * built against a newer header checks SPECFEM_HAS_FIELD before reading a
 * field the writer may not know. Arrays of extensible structs are passed as
 * arrays of pointers so that their stride never depends on struct_size.
 *
 * Plugin authors writing C++ should use specfem/plugin/injection_plugin.hpp,
 * which implements this table on top of an abstract base class.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Version of the injection plugin ABI defined by this header. */
#define SPECFEM_INJECTION_ABI_VERSION 1

/** @brief Version of the @ref specfem_manifest layout. */
#define SPECFEM_MANIFEST_VERSION 1

/** @brief Manifest @c plugin_kind value for wavefield-injection plugins. */
#define SPECFEM_PLUGIN_KIND_INJECTION 1

/** @brief Name of the single symbol every injection plugin exports. */
#define SPECFEM_INJECTION_PLUGIN_ENTRY_NAME "specfem_injection_plugin"

#if defined(_WIN32)
#define SPECFEM_EXPORT __declspec(dllexport)
#else
/** @brief Marks the plugin entry point as visible from outside the library. */
#define SPECFEM_EXPORT __attribute__((visibility("default")))
#endif

/**
 * @brief Whether an extensible struct contains a field.
 *
 * @param PTR Pointer to the struct
 * @param TYPE Struct type
 * @param FIELD Field name
 */
#define SPECFEM_HAS_FIELD(PTR, TYPE, FIELD)                                    \
  ((PTR)->struct_size >= offsetof(TYPE, FIELD) + sizeof(((TYPE *)0)->FIELD))

/** @brief Status code returned by plugin functions (see
 * specfem_status_values). */
typedef int32_t specfem_status;

/** @brief Values of @c specfem_status. */
enum specfem_status_values {
  specfem_status_ok = 0,         /**< Success */
  specfem_status_error = 1,      /**< Failure; details via last_error */
  specfem_status_unsupported = 2 /**< The plugin cannot serve this run;
                                        details via last_error. SPECFEM++
                                        stops with that message and does not
                                        retry. */
};

/** @brief Scalar type of a buffer (see specfem_dtype_values). */
typedef int32_t specfem_dtype;

/** @brief Values of @c specfem_dtype. */
enum specfem_dtype_values {
  specfem_dtype_float32 = 0, /**< 32-bit IEEE float */
  specfem_dtype_float64 = 1  /**< 64-bit IEEE float */
};

/**
 * @brief Medium of a point group (see specfem_medium_values).
 *
 * Values are the medium tag identifiers of SPECFEM++ and never change.
 */
typedef int32_t specfem_medium;

/** @brief Values of @c specfem_medium available in three dimensions. */
enum specfem_medium_values {
  specfem_medium_acoustic = 3,    /**< Acoustic fluid */
  specfem_medium_elastic = 6,     /**< Elastic solid */
  specfem_medium_elastic_spin = 7 /**< Elastic solid with spin (Cosserat) */
};

/**
 * @brief Material parameterization of a point group (see
 * specfem_property_values).
 *
 * Values are the property tag identifiers of SPECFEM++ and never change.
 */
typedef int32_t specfem_property;

/** @brief Values of @c specfem_property. */
enum specfem_property_values {
  specfem_property_isotropic = 0,         /**< Isotropic */
  specfem_property_anisotropic = 1,       /**< Anisotropic (full tensor) */
  specfem_property_isotropic_cosserat = 2 /**< Isotropic Cosserat */
};

/** @brief Physical quantity requested from a plugin (see
 * specfem_quantity_values). */
typedef int32_t specfem_quantity;

/**
 * @brief Values of @c specfem_quantity.
 *
 * The number of buffer components a quantity occupies depends on the medium
 * and is sent by SPECFEM++ in @ref specfem_quantity_slot; plugins never
 * derive it.
 */
enum specfem_quantity_values {
  specfem_quantity_displacement = 0,  /**< Displacement */
  specfem_quantity_velocity = 1,      /**< Velocity */
  specfem_quantity_acceleration = 2,  /**< Acceleration */
  specfem_quantity_traction = 3,      /**< Traction on the point's normal */
  specfem_quantity_potential_rate = 4 /**< Fluid potential rate */
};

/**
 * @brief MPI communicator handed to the plugin.
 *
 * The communicator is passed as its Fortran handle (@c MPI_Comm_c2f) so this
 * header does not depend on @c mpi.h. A plugin that uses MPI converts it back
 * with @c MPI_Comm_f2c and must use the same MPI implementation as SPECFEM++.
 */
typedef struct specfem_communicator {
  uint32_t struct_size;   /**< sizeof(specfem_communicator) */
  int32_t has_mpi;        /**< 0 when SPECFEM++ runs without MPI */
  int32_t rank;           /**< Rank of this process in the communicator */
  int32_t size;           /**< Number of ranks in the communicator */
  int32_t fortran_handle; /**< MPI_Comm_c2f(comm); unused when has_mpi == 0 */
} specfem_communicator;

/** @brief Simulation parameters shared with the plugin at creation. */
typedef struct specfem_simulation_info {
  uint32_t struct_size; /**< sizeof(specfem_simulation_info) */
  int32_t dimension;    /**< Spatial dimension; 3 in ABI v1 */
  int32_t nstep;        /**< Total number of time steps */
  int32_t reserved;     /**< Must be 0 */
  double t0;            /**< Time of step 0 */
  double dt;            /**< Time step */
  /** Bounding box of the global domain (the whole mesh, not the rank-local
   *  part), identical on every rank: xmin, xmax, ymin, ymax, zmin, zmax. */
  double bounds[6];
} specfem_simulation_info;

/**
 * @brief Layout of one requested quantity in a group's buffer.
 *
 * Fixed layout (not extensible): passed as a plain array.
 */
typedef struct specfem_quantity_slot {
  specfem_quantity quantity; /**< Requested quantity */
  int32_t offset;            /**< First buffer component of the quantity */
  int32_t count;             /**< Number of components of the quantity */
  int32_t reserved;          /**< Must be 0 */
} specfem_quantity_slot;

/**
 * @brief Points of one material system and the quantities requested there.
 *
 * The group covers points [begin, begin + count) of @ref specfem_points.
 * Material properties are named after the fields of the SPECFEM++ point
 * property container of the system and stored point-major:
 * property k of local point p is @c properties[p * nproperties + k]. For
 * example, elastic/isotropic provides @c kappa, @c mu and @c rho; plugins
 * should look properties up by name rather than by position.
 */
typedef struct specfem_point_group {
  uint32_t struct_size;              /**< sizeof(specfem_point_group) */
  specfem_medium medium;             /**< Medium of the group */
  specfem_property property;         /**< Material parameterization */
  int32_t nproperties;               /**< Number of material properties */
  int64_t begin;                     /**< First point of the group */
  int64_t count;                     /**< Number of points (may be 0) */
  const char *const *property_names; /**< [nproperties] property names */
  const double *properties; /**< [count * nproperties] property values */
  int32_t ncomponents;      /**< Buffer components per point */
  int32_t nslots;           /**< Number of requested quantities */
  const specfem_quantity_slot *slots; /**< [nslots] requested quantities */
} specfem_point_group;

/**
 * @brief Points at which the incident field is requested.
 *
 * Host arrays, read-only, valid only during @c setup. Normals are outward unit
 * normals of the boundary the point lies on. Points are ordered by group, and
 * the groups partition [0, n) in order.
 */
typedef struct specfem_points {
  uint32_t struct_size; /**< sizeof(specfem_points) */
  int32_t ngroups;      /**< Number of groups (same on every rank) */
  int64_t n;            /**< Number of points */
  const double *x;      /**< x-coordinates */
  const double *y;      /**< y-coordinates */
  const double *z;      /**< z-coordinates (up) */
  const double *nx;     /**< Unit normal, x-component */
  const double *ny;     /**< Unit normal, y-component */
  const double *nz;     /**< Unit normal, z-component */
  const specfem_point_group *const *groups; /**< [ngroups] point groups */
} specfem_points;

/**
 * @brief Host buffer the plugin fills for one group and window of time steps.
 *
 * Element (point p, component c, step s) of the window lives at
 * @c data[p * stride_point + c * stride_component + s * stride_step], with
 * strides counted in elements of @c dtype. Point p is point p of the group;
 * step s is simulation step @c first_step + s of the @c fill call.
 */
typedef struct specfem_buffer {
  uint32_t struct_size;     /**< sizeof(specfem_buffer) */
  specfem_dtype dtype;      /**< Element type */
  void *data;               /**< Host memory to write */
  int64_t npoints;          /**< Points; equals the group's count */
  int32_t ncomponents;      /**< Components; equals the group's ncomponents */
  int32_t nsteps;           /**< Steps in this window */
  int64_t stride_point;     /**< Stride between points */
  int64_t stride_component; /**< Stride between components */
  int64_t stride_step;      /**< Stride between steps */
} specfem_buffer;

/**
 * @brief Function table describing an injection plugin.
 *
 * Returned by the plugin's exported @c specfem_injection_plugin(). All
 * function pointers are required.
 */
typedef struct specfem_injection_plugin_table {
  uint32_t struct_size;  /**< sizeof(specfem_injection_plugin_table) */
  uint32_t abi_version;  /**< Must equal SPECFEM_INJECTION_ABI_VERSION */
  uint64_t capabilities; /**< Capability flags; 0 in ABI v1 */
  const char *name;      /**< Plugin name */
  const char *version;   /**< Plugin version string */

  /** Create a plugin instance; returns NULL on failure (see last_error). */
  void *(*create)(const char *config_path, const specfem_simulation_info *info,
                  const specfem_communicator *communicator);

  /** Return 1 if the plugin can provide @p quantity for points of the given
   *  medium and property, else 0. SPECFEM++ queries the global set of
   *  systems, so the calls are the same on every rank. */
  int32_t (*supports)(void *self, specfem_medium medium,
                      specfem_property property, specfem_quantity quantity);

  /** Receive the injection points, their groups and the request (once). */
  specfem_status (*setup)(void *self, const specfem_points *points);

  /** Fill steps [first_step, first_step + nsteps) of every group: buffer g
   *  belongs to group g, and nbuffers equals the number of groups. */
  specfem_status (*fill)(void *self, int32_t first_step, int32_t nbuffers,
                         const specfem_buffer *const *buffers);

  /** Last error message of @p self; with NULL, the last failed create. */
  const char *(*last_error)(const void *self);

  /** Destroy an instance returned by create (NULL is a no-op). */
  void (*destroy)(void *self);
} specfem_injection_plugin_table;

/** @brief Signature of the exported plugin entry point. */
typedef const specfem_injection_plugin_table *(*specfem_injection_plugin_entry)(
    void);

/**
 * @brief Manifest embedded in every plugin, readable without loading it.
 *
 * SPECFEM++ scans the plugin file for the magic bytes and checks the ABI
 * version before calling @c dlopen, so a mismatched plugin never runs code in
 * the simulation process.
 */
typedef struct specfem_manifest {
  char magic[16];            /**< SPECFEM_MANIFEST_MAGIC_INITIALIZER bytes */
  uint32_t manifest_version; /**< SPECFEM_MANIFEST_VERSION */
  uint32_t abi_version;      /**< SPECFEM_INJECTION_ABI_VERSION */
  uint32_t plugin_kind;      /**< SPECFEM_PLUGIN_KIND_INJECTION */
  uint32_t reserved;         /**< Must be 0 */
  char name[64];             /**< NUL-terminated plugin name */
  char version[32];          /**< NUL-terminated plugin version */
} specfem_manifest;

/*
 * The magic is spelled out byte by byte so that it never appears in a plugin
 * as a string literal: the manifest must be the only place a scanner finds it.
 */
#define SPECFEM_MANIFEST_MAGIC_INITIALIZER                                     \
  { 'S', 'P', 'E', 'C', 'F', 'E', 'M', 'P',                                    \
    'P', '_', 'P', 'L', 'U', 'G', 'I', 'N' }

#if defined(__APPLE__)
#define SPECFEM_MANIFEST_SECTION "__DATA,__specfem"
#else
#define SPECFEM_MANIFEST_SECTION ".specfem_plugin"
#endif

#if defined(__has_attribute) && !defined(__APPLE__)
#if __has_attribute(retain)
#define SPECFEM_MANIFEST_RETAIN __attribute__((retain))
#endif
#endif
#ifndef SPECFEM_MANIFEST_RETAIN
#define SPECFEM_MANIFEST_RETAIN
#endif

/**
 * @brief Define the plugin manifest (once per plugin, at file scope).
 *
 * @param NAME    String literal, at most 63 characters.
 * @param VERSION String literal, at most 31 characters.
 */
#define SPECFEM_DEFINE_INJECTION_MANIFEST(NAME, VERSION)                       \
  __attribute__((used, section(SPECFEM_MANIFEST_SECTION)))                     \
  SPECFEM_MANIFEST_RETAIN static const specfem_manifest                        \
      specfem_injection_manifest = { SPECFEM_MANIFEST_MAGIC_INITIALIZER,       \
                                     SPECFEM_MANIFEST_VERSION,                 \
                                     SPECFEM_INJECTION_ABI_VERSION,            \
                                     SPECFEM_PLUGIN_KIND_INJECTION,            \
                                     0u,                                       \
                                     NAME,                                     \
                                     VERSION }

/** @brief The plugin entry point; every injection plugin defines it. */
SPECFEM_EXPORT const specfem_injection_plugin_table *
specfem_injection_plugin(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
