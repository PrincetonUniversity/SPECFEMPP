#ifndef SPECFEM_MPI_H
#define SPECFEM_MPI_H

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace specfem {
class MPI {
public:
  MPI(int *argc, char ***argv);
  void sync_all();
  int get_size();
  int get_rank();
  void exit();
  ~MPI();

private:
  int world_size, my_rank;
#ifdef MPI_PARALLEL
  MPI_Comm comm;
#endif
};
} // namespace specfem

#endif // SPECFEM_MPI_H
