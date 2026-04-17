#pragma once

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#ifdef SPECFEM_ENABLE_MPI
#include <mpi.h>
#endif

// Non-MPI shims: provide MPI type aliases and sentinel constants so that
// headers declaring MPI-typed members/parameters compile without <mpi.h>.
#ifndef SPECFEM_ENABLE_MPI

// --- Handle types (all opaque handles are integers in MPICH convention) ---
using MPI_Comm = int;
using MPI_Request = int;
using MPI_Datatype = int;
using MPI_Op = int;
using MPI_Info = int;
using MPI_Group = int;
using MPI_Win = int;
using MPI_Errhandler = int;
using MPI_File = int;
using MPI_Message = int;

// --- MPI_Status (struct — user code accesses .MPI_SOURCE, .MPI_TAG, etc.) ---
struct MPI_Status {
  int MPI_SOURCE;
  int MPI_TAG;
  int MPI_ERROR;
};

// --- Null sentinel constants for handle types ---
constexpr MPI_Comm MPI_COMM_WORLD = 0;
constexpr MPI_Comm MPI_COMM_NULL = -1;
constexpr MPI_Comm MPI_COMM_SELF = 1;
constexpr MPI_Request MPI_REQUEST_NULL = -1;
constexpr MPI_Datatype MPI_DATATYPE_NULL = -1;
constexpr MPI_Op MPI_OP_NULL = -1;
constexpr MPI_Info MPI_INFO_NULL = -1;
constexpr MPI_Group MPI_GROUP_NULL = -1;
constexpr MPI_Group MPI_GROUP_EMPTY = -2;
constexpr MPI_Win MPI_WIN_NULL = -1;
constexpr MPI_Errhandler MPI_ERRHANDLER_NULL = -1;
constexpr MPI_File MPI_FILE_NULL = -1;
constexpr MPI_Message MPI_MESSAGE_NULL = -1;

// --- Status pointer sentinels ---
#define MPI_STATUS_IGNORE (static_cast<MPI_Status *>(nullptr))
#define MPI_STATUSES_IGNORE (static_cast<MPI_Status *>(nullptr))

#endif

namespace specfem {

// Forward declaration
namespace program {
class Context;
[[noreturn]]
void abort(const std::string &message, int error_code, const int line,
           const char *file);
} // namespace program

/**
 * @class MPI
 * @brief Static MPI wrapper for SPECFEM++
 *
 * This class provides a static interface to MPI functionality,
 * eliminating the need to pass MPI pointers throughout the codebase.
 *
 * Key features:
 * - Static rank and size members accessible globally
 * - Context-managed lifecycle (only Context can initialize/finalize)
 * - Safety checks to prevent use outside Context scope
 * - Minimal API focused on essential MPI operations
 *
 * Usage:
 * @code
 * // After Context is initialized
 * int my_rank = specfem::MPI::get_rank();
 * int world_size = specfem::MPI::get_size();
 * specfem::MPI::sync();
 * @endcode
 *
 * @note This class cannot be instantiated. All members are static.
 * @note Only specfem::program::Context can initialize/finalize this class.
 */
class MPI {

private:
  static int rank_; ///< Current MPI rank (-1 if not initialized)
  static int size_; ///< Total number of MPI processes (-1 if not initialized)
  static MPI_Comm comm_; ///< MPI communicator (MPI_COMM_WORLD or user-defined)

public:
  /**
   * @brief Synchronize all MPI processes (MPI_Barrier)
   *
   * @throws Calls MPI_Abort if called by an excluded rank
   */
  static void sync() {
    if (!check_context()) {
      abort_excluded_rank("sync() called on excluded rank");
    }
#ifdef SPECFEM_ENABLE_MPI
    MPI_Barrier(comm_);
#endif
  }

  /**
   * @brief Synchronize all MPI processes (alias for sync())
   *
   * @throws Exits with error code 1 if called outside Context scope
   */
  static void sync_all() { sync(); }

  /**
   * @brief Get MPI rank
   *
   * @return int Current MPI rank
   * @throws Exits with error code 1 if called outside Context scope
   */
  static int get_rank() {
    if (!check_context()) {
      abort_excluded_rank("get_rank() called on excluded rank");
    }
    return rank_;
  }

  /**
   * @brief Get MPI world size
   *
   * @return int Total number of MPI processes
   * @throws Exits with error code 1 if called outside Context scope
   */
  static int get_size() {
    if (!check_context()) {
      abort_excluded_rank("get_size() called on excluded rank");
    }
    return size_;
  }

  /**
   * @brief Get active communicator for simulations
   *
   * Returns the active communicator, which may be a subset communicator
   * (if initialized with nprocs) or MPI_COMM_WORLD (default initialization).
   * For excluded ranks in subset initialization, returns MPI_COMM_NULL.
   *
   * @return MPI_Comm The active communicator for simulations
   * @throws Exits with error code 1 if called outside Context scope
   */
  static MPI_Comm communicator() {
    if (!check_context()) {
      abort_excluded_rank("communicator() called on excluded rank");
    }
    return comm_;
  }

  /**
   * @brief Check if current process is the main process (rank 0)
   *
   * @return bool True if rank == 0
   * @throws Exits with error code 1 if called outside Context scope
   */
  static bool main_proc() {
    if (!check_context()) {
      abort_excluded_rank("main_proc() called on excluded rank");
    }
    return rank_ == 0;
  }

  /**
   * @brief Format filename with processor number using dynamic zero-padding
   *
   * For multi-process runs, transforms "dir/stem.ext" into
   * "dir/stem/proc_N.ext" where N is the zero-padded rank.
   * For single-process runs, the filename is returned unchanged.
   *
   * Examples (size=6, rank=2):
   * - "foo/bar.bin" -> "foo/bar/proc_2.bin"
   * - "foo/bar.bin" (size=100, rank=2) -> "foo/bar/proc_02.bin"
   * - "bar.bin"     -> "bar/proc_2.bin"
   *
   * @param filename Input filename (can include directory path)
   * @return std::string Formatted filename with processor number
   * @throws Exits with error code 1 if called outside Context scope
   */
  /**
   * @brief Check if this rank is part of the active communicator
   *
   * Returns false for ranks excluded by subset initialization
   * (i.e., ranks >= nprocs when initialized with nprocs).
   * Safe to call on any rank without triggering exit.
   *
   * @return bool True if this rank holds a valid communicator
   */
  static bool is_active() { return comm_ != MPI_COMM_NULL; }

  static bool check_context() {
    // Excluded ranks (subset initialization) have comm_ == MPI_COMM_NULL
    // but are validly initialized — return false instead of aborting.
    if (comm_ == MPI_COMM_NULL) {
      return false;
    }
    if (rank_ == -1 || size_ == -1) {
      std::cerr << "ERROR: MPI used outside Context scope" << std::endl;
      std::exit(1);
    }
    return true;
  }

  static std::string format_proc_filename(const std::string &filename) {
    if (!check_context()) {
      abort_excluded_rank("format_proc_filename() called on excluded rank");
    }

    // For single process, return filename unchanged
    if (size_ <= 1) {
      return filename;
    }

    // Calculate number of digits needed for processor numbering
    int ndigits = static_cast<int>(std::log10(size_ - 1)) + 1;

    // Format processor number with zero-padding
    std::ostringstream proc_str;
    proc_str << std::setfill('0') << std::setw(ndigits) << rank_;

    // Use std::filesystem for cross-platform path handling
    std::filesystem::path p(filename);
    auto stem = p.stem();
    auto ext = p.extension();
    auto parent = p.parent_path();

    // New scheme: dir/stem/proc_N.ext
    std::filesystem::path result =
        parent / stem / ("proc_" + proc_str.str() + ext.string());
    return result.string();
  }

private:
  MPI() = default;
  ~MPI() = default;
  MPI(const MPI &) = delete;
  MPI &operator=(const MPI &) = delete;

  /**
   * @brief Initialize MPI and set rank/size
   *
   * Called by Context constructor. Checks if MPI is already initialized
   * externally before calling MPI_Init.
   *
   * @param argc Pointer to argument count
   * @param argv Pointer to argument vector
   */
  static void initialize(int *argc, char ***argv);

  /**
   * @brief Initialize MPI with user-defined processes within world communicator
   *
   * Creates a subset MPI communicator containing only the first nprocs ranks.
   * Ranks >= nprocs receive MPI_COMM_NULL and are excluded from simulation.
   *
   * @param argc Pointer to argument count
   * @param argv Pointer to argument vector
   * @param nprocs Number of processes to include in the new communicator (must
   * be
   * <= world size)
   * @throws Exits with error code 1 if nprocs > world size
   */
  static void initialize(int *argc, char ***argv, int nprocs);

  /**
   * @brief Finalize MPI and reset rank/size to -1
   *
   * Called by Context destructor. Only calls MPI_Finalize if MPI
   * was initialized by this wrapper (not externally).
   */
  static void finalize();

  [[noreturn]] static void abort_excluded_rank(const std::string &message) {
#ifdef SPECFEM_ENABLE_MPI
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "ERROR on rank " << rank << ": " << message << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    std::cerr << "ERROR: " << message << std::endl;
    std::exit(1);
#endif
  }

  friend class specfem::program::Context;
  friend void specfem::program::abort(const std::string &, int, const int,
                                      const char *);
};

} // namespace specfem

#ifndef SPECFEM_ENABLE_MPI
#define SPECFEM_MPI_SAFECALL(call) ((void)0)
#else
#define SPECFEM_MPI_SAFECALL(call)                                             \
  do {                                                                         \
    specfem::MPI::check_context();                                             \
    int e = call;                                                              \
    if (e != MPI_SUCCESS) {                                                    \
      char err[MPI_MAX_ERROR_STRING];                                          \
      int len;                                                                 \
      MPI_Error_string(e, err, &len);                                          \
      std::cerr << "MPI error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << err << std::endl;                                           \
      MPI_Abort(MPI_COMM_WORLD, e);                                            \
    }                                                                          \
  } while (0)
#endif

#ifndef SPECFEM_ENABLE_MPI
#define SPECFEM_MPI_ON_ROOT(code)                                              \
  do {                                                                         \
    code                                                                       \
  } while (0)
#else
#define SPECFEM_MPI_ON_ROOT(code)                                              \
  do {                                                                         \
    if (specfem::MPI::main_proc()) {                                           \
      code                                                                     \
    }                                                                          \
  } while (0)
#endif
