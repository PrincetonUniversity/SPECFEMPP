#ifndef SPECFEM_MPI_H
#define SPECFEM_MPI_H

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace specfem {

/**
 * @brief MPI class instance to manage MPI communication
 *
 */

/**
 * @note If specfem is compiled without MPI then world_size = 1 and my_rank = 0
 * Additionally, many routines are just empty to optimize performance.
 *
 */

class MPI {
public:
  /**
   * @brief Initialize a MPI object
   */
  MPI(int *argc, char ***argv);
  /**
   * @brief Sync all process. MPI_Barrier
   *
   */
  void sync_all();
  /**
   * @brief Get world_size
   *
   * @return int world size
   */
  int get_size();
  /**
   * @brief Get my_rank
   *
   * @return int my_rank
   */
  int get_rank();
  /**
   * @brief MPI_Finalize
   *
   */
  void exit();
  ~MPI();

private:
  int world_size, my_rank;
#ifdef MPI_PARALLEL
  static bool MPI_initialized = false;
  MPI_Comm comm;
#endif
};
} // namespace specfem

#endif // SPECFEM_MPI_H
