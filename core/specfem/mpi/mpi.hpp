#pragma once

#include <cstdlib>
#include <iostream>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace specfem {

#ifdef MPI_PARALLEL
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
 * int my_rank = specfem::MPI::MPI::rank;
 * int world_size = specfem::MPI::MPI::size;
 * specfem::MPI::MPI::sync();
 * @endcode
 *
 * @note This class cannot be instantiated. All members are static.
 * @note Only specfem::program::Context can initialize/finalize this class.
 */
class MPI_new {
public:
  static int rank; ///< Current MPI rank (-1 if not initialized)
  static int size; ///< Total number of MPI processes (-1 if not initialized)

  /**
   * @brief Synchronize all MPI processes (MPI_Barrier)
   *
   * @throws Exits with error code 1 if called outside Context scope
   */
  static void sync() {
    check_context();
#ifdef MPI_PARALLEL
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
    return rank;
  }

  /**
   * @brief Get MPI world size
   *
   * @return int Total number of MPI processes
   * @throws Exits with error code 1 if called outside Context scope
   */
  static int get_size() {
    check_context();
    return size;
  }

  /**
   * @brief Check if current process is the main process (rank 0)
   *
   * @return bool True if rank == 0
   * @throws Exits with error code 1 if called outside Context scope
   */
  static bool main_proc() {
    check_context();
    return rank == 0;
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
#ifdef MPI_PARALLEL
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
#ifdef MPI_PARALLEL
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
#ifdef MPI_PARALLEL
    MPI_Reduce(&lvalue, &result, 1, MPI_DOUBLE, reduce_op, 0, MPI_COMM_WORLD);
#endif
    return result;
  }

private:
  MPI_new() = default;
  ~MPI_new() = default;
  MPI_new(const MPI_new &) = delete;
  MPI_new &operator=(const MPI_new &) = delete;

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
    if (rank == -1 || size == -1) {
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
