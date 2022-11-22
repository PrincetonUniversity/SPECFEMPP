#ifndef SPECFEM_MPI_H
#define SPECFEM_MPI_H

#include <iostream>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace specfem {

namespace MPI {
#ifdef MPI_PARALLEL
/**
 * @brief MPI reducer type
 *
 * Incase specfem is compiled without MPI then I need placeholders for reducer
 * types
 */
enum reduce_type { sum : MPI_SUM, min : MPI_MIN, max : MPI_MAX };
#endif
/**
 * @brief MPI reducer type
 *
 */
enum reduce_type { sum, min, max };

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
   * @param reduce_type specfem reducer type
   * @return int Reduced value. Should only be reduced on the root=0 process.
   */
  int reduce(int lvalue, specfem::MPI::reduce_type reduce_type) const;
  /**
   * @brief MPI reduce implemetation
   *
   * @param lvalue local value to reduce
   * @param reduce_type specfem reducer type
   * @return int Reduced value. Should only be reduced on the root=0 process.
   */
  float reduce(float lvalue, specfem::MPI::reduce_type reduce_type) const;
  /**
   * @brief MPI reduce implemetation
   *
   * @param lvalue local value to reduce
   * @param reduce_type specfem reducer type
   * @return int Reduced value. Should only be reduced on the root=0 process.
   */
  double reduce(double lvalue, specfem::MPI::reduce_type reduce_type) const;
  /**
   * @brief MPI all reduce implementation
   *
   * @param lvalue local value to reduce
   * @param reduce_type
   * @return int Reduced value. Should only be reduced on the root=0 process.
   */
  int all_reduce(int lvalue, specfem::MPI::reduce_type reduce_type) const;
  /**
   * @brief MPI all reduce implementation
   *
   * @param lvalue local value to reduce
   * @param reduce_type
   * @return int Reduced value. Should only be reduced on the root=0 process.
   */
  float all_reduce(float lvalue, specfem::MPI::reduce_type reduce_type) const;
  /**
   * @brief MPI all reduce implementation
   *
   * @param lvalue local value to reduce
   * @param reduce_type
   * @return int Reduced value. Should only be reduced on the root=0 process.
   */
  double all_reduce(double lvalue, specfem::MPI::reduce_type reduce_type) const;

private:
  int world_size; ///< total number of MPI processes
  int my_rank;    ///< rank of my process
#ifdef MPI_PARALLEL
  MPI_Comm comm; ///< MPI communicator
#endif
};
} // namespace MPI

} // namespace specfem

#endif // SPECFEM_MPI_H
