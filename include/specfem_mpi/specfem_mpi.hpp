#ifndef _SPECFEM_MPI_HPP
#define _SPECFEM_MPI_HPP

#include <iostream>
#include <vector>

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
using reduce_type = MPI_Op;
const static reduce_type sum = MPI_SUM;
const static reduce_type min = MPI_MIN;
const static reduce_type max = MPI_MAX;
#else
/**
 * @brief MPI reducer type
 *
 */
enum reduce_type { sum, min, max };
#endif

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
   * @brief Checks if current proc is main proc
   *
   * @return bool rank of the main proc
   */
  bool main_proc() const { return this->get_rank() == this->get_main(); };
  /**
   * @brief Gets rank of main proc.
   *
   * For now rank = 0 is hard coded as the main proc
   *
   * @return int rank of the main proc
   */
  int get_main() const { return 0; }
  /**
   * @brief MPI_Abort
   *
   */
  void exit();
  /**
   * @brief Print string s from the head node
   *
   */
  template <typename T> void cout(T s, bool root_only = true) const;

  /**
   * @brief Print a formatted header with SPECFEM++ branding
   *
   * @param s Header string to print (default: "SPECFEM++")
   */
  void print_header(const std::string &s = "SPECFEM++") const;

  /**
   * @brief Print a title string with surrounding decoration
   *
   * @param s Title string to print
   */
  void print_title(const std::string &s) const;

  /**
   * @brief Print a string (wrapper for cout with root_only=true)
   *
   * @param s String to print
   */
  template <typename T> void print(T s) const;

  /**
   * @brief Destroy the MPI object
   *
   */
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

  /**
   * @brief Gathers elements from all procs in communicator in a vector on main
   *
   * @param lelement element to gather
   * @return std::vector<int> vector of gathered elements
   */
  std::vector<int> gather(int lelement) const;
  /**
   * @brief Gathers elements from all procs in communicator in a vector on main
   *
   * @param lelement element to gather
   * @return std::vector<float> vector of gathered elements
   */
  std::vector<float> gather(float lelement) const;
  /**
   * @brief Gathers elements from all procs in communicator in a vector on main
   *
   * @param lelement element to gather
   * @return std::vector<double> vector of gathered elements
   */
  std::vector<double> gather(double lelement) const;

  /**
   * @brief scatter elements on main proc to rest of the processors
   *
   * @param gelement vector of elements to scatter
   * @return int scattered element
   */
  int scatter(std::vector<int> gelement) const;
  /**
   * @brief scatter elements on main proc to rest of the processors
   *
   * @param gelement vector of elements to scatter
   * @return float scattered element
   */
  float scatter(std::vector<float> gelement) const;
  /**
   * @brief scatter elements on main proc to rest of the processors
   *
   * @param gelement vector of elements to scatter
   * @return double scattered element
   */
  double scatter(std::vector<double> gelement) const;

  /**
   * @brief Broadcast a value from main proc to the rest
   *
   * @param val value to broadcast
   */
  void bcast(int &val) const;
  /**
   * @brief Broadcast a value from main proc to the rest
   *
   * @param val value to broadcast
   */
  void bcast(float &val) const;
  /**
   * @brief Broadcast a value from main proc to the rest
   *
   * @param val value to broadcast
   */
  void bcast(double &val) const;

  /**
   * @brief Broadcast a value from root proc to the rest
   *
   * @param val value to broadcast
   */
  void bcast(int &val, int root) const;
  /**
   * @brief Broadcast a value from root proc to the rest
   *
   * @param val value to broadcast
   */
  void bcast(float &val, int root) const;
  /**
   * @brief Broadcast a value from root proc to the rest
   *
   * @param val value to broadcast
   */
  void bcast(double &val, int root) const;

private:
  int world_size;  ///< total number of MPI processes
  int my_rank;     ///< rank of my process
  int extern_init; ///< flag to check if MPI was initialized outside SPECFEM
#ifdef MPI_PARALLEL
  MPI_Comm comm; ///< MPI communicator
#endif
};
} // namespace MPI

} // namespace specfem

#endif // SPECFEM_MPI_H
