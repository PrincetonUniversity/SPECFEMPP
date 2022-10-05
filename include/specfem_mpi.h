#ifndef SPECFEM_MPI_H
#define SPECFEM_MPI_H

#include <iostream>

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
  void sync_all() const;
  /**
   * @brief Get world_size
   *
   * @return int world size
   */
  int get_size() const;
  /**
   * @brief Get my_rank
   *
   * @return int my_rank
   */
  int get_rank() const;
  /**
   * @brief MPI_Abort
   *
   */
  void exit();
  /**
   * @brief Print string s from the head node
   *
   */
  template <typename T> void cout(T s) const {
#ifdef MPI_PARALLEL
    if (my_rank == 0) {
      std::cout << s << std::endl;
    }
#else
    std::cout << s << std::endl;
#endif
  }

  ~MPI();

  /**
   * @brief MPI reduce implemetation
   *
   * @param lvalue local value to reduce
   * @return int Reduced value. Should only be reduced on the root=0 process.
   */
  int reduce(int lvalue) const;

private:
  int world_size, my_rank;
#ifdef MPI_PARALLEL
  MPI_Comm comm;
#endif
};
} // namespace specfem

#endif // SPECFEM_MPI_H
