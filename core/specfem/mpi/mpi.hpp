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

public:
  /**
   * @brief Synchronize all MPI processes (MPI_Barrier)
   *
   * @throws Exits with error code 1 if called outside Context scope
   */
  static void sync() {
    check_context();
#ifdef SPECFEM_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
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
    check_context();
    return rank_;
  }

  /**
   * @brief Get MPI world size
   *
   * @return int Total number of MPI processes
   * @throws Exits with error code 1 if called outside Context scope
   */
  static int get_size() {
    check_context();
    return size_;
  }

  /**
   * @brief Check if current process is the main process (rank 0)
   *
   * @return bool True if rank == 0
   * @throws Exits with error code 1 if called outside Context scope
   */
  static bool main_proc() {
    check_context();
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
  static bool check_context() {
    if (rank_ == -1 || size_ == -1) {
      std::cerr << "ERROR: MPI used outside Context scope" << std::endl;
      std::exit(1);
    }
    return true;
  }

  static std::string format_proc_filename(const std::string &filename) {
    check_context();

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
   * @brief Finalize MPI and reset rank/size to -1
   *
   * Called by Context destructor. Only calls MPI_Finalize if MPI
   * was initialized by this wrapper (not externally).
   */
  static void finalize();

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
      fprintf(stderr, "MPI error %s:%d: %s\n", __FILE__, __LINE__, err);       \
      MPI_Abort(MPI_COMM_WORLD, e);                                            \
    }                                                                          \
  } while (0)
#endif
