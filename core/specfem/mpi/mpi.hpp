#pragma once

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#ifdef SPECFEM_ENABLE_MPI
#include <mpi.h>
#endif

namespace specfem {

#ifdef SPECFEM_ENABLE_MPI
using reduce_type = MPI_Op;
const static reduce_type sum = MPI_SUM;
const static reduce_type min = MPI_MIN;
const static reduce_type max = MPI_MAX;
#else
enum reduce_type { sum, min, max };
#endif

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
   * Inserts processor number into filename with zero-padding based on total
   * number of processes. For example:
   * - "foo/bar.bin" with size=4  -> "foo/proc0_bar.bin"
   * - "foo/bar.bin" with size=100 -> "foo/proc00_bar.bin"
   * - "bar.bin" with size=1000 -> "proc000_bar.bin"
   *
   * @param filename Input filename (can include directory path)
   * @return std::string Formatted filename with processor number
   * @throws Exits with error code 1 if called outside Context scope
   */
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

    // Find last slash to split directory and filename
    size_t lastslash = filename.find_last_of('/');

    if (lastslash != std::string::npos) {
      // Split into directory and filename
      std::string dir_path = filename.substr(0, lastslash + 1);
      std::string file_name = filename.substr(lastslash + 1);
      // Construct: dir_path/proc###_filename
      return dir_path + "proc" + proc_str.str() + "_" + file_name;
    } else {
      // No directory path, just prepend proc###_
      return "proc" + proc_str.str() + "_" + filename;
    }
  }

  /**
   * @brief MPI reduce operation
   *
   * @param lvalue Local value to reduce
   * @param reduce_op Reduction operation (specfem::sum, specfem::min,
   * specfem::max)
   * @return Reduced value (only valid on root process)
   * @throws Exits with error code 1 if called outside Context scope
   */
  static int reduce(int lvalue, specfem::reduce_type reduce_op) {
    check_context();
    int result = lvalue;
#ifdef SPECFEM_ENABLE_MPI
    MPI_Reduce(&lvalue, &result, 1, MPI_INT, reduce_op, 0, MPI_COMM_WORLD);
#endif
    return result;
  }

  /**
   * @brief MPI reduce operation for float
   *
   * @param lvalue Local value to reduce
   * @param reduce_op Reduction operation
   * @return Reduced value (only valid on root process)
   * @throws Exits with error code 1 if called outside Context scope
   */
  static float reduce(float lvalue, specfem::reduce_type reduce_op) {
    check_context();
    float result = lvalue;
#ifdef SPECFEM_ENABLE_MPI
    MPI_Reduce(&lvalue, &result, 1, MPI_FLOAT, reduce_op, 0, MPI_COMM_WORLD);
#endif
    return result;
  }

  /**
   * @brief MPI reduce operation for double
   *
   * @param lvalue Local value to reduce
   * @param reduce_op Reduction operation
   * @return Reduced value (only valid on root process)
   * @throws Exits with error code 1 if called outside Context scope
   */
  static double reduce(double lvalue, specfem::reduce_type reduce_op) {
    check_context();
    double result = lvalue;
#ifdef SPECFEM_ENABLE_MPI
    MPI_Reduce(&lvalue, &result, 1, MPI_DOUBLE, reduce_op, 0, MPI_COMM_WORLD);
#endif
    return result;
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

  /**
   * @brief Check if MPI is initialized (Context exists)
   *
   * Verifies that rank and size are valid (not -1).
   * Exits with error code 1 if check fails.
   */
  static bool check_context() {
    if (rank_ == -1 || size_ == -1) {
      std::cerr << "ERROR: MPI used outside Context scope" << std::endl;
      std::exit(1);
    }
    return true;
  }

  friend class specfem::program::Context;
  friend void specfem::program::abort(const std::string &, int, const int,
                                      const char *);
};

} // namespace specfem
