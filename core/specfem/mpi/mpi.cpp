#include "mpi.hpp"

namespace specfem {

// Static storage for MPI rank and size (-1 indicates uninitialized)
int MPI::rank_ = -1;
int MPI::size_ = -1;

void MPI::initialize(int *argc, char ***argv) {
#ifdef SPECFEM_ENABLE_MPI
  int initialized;
  MPI_Initialized(&initialized);

  if (!initialized) {
    MPI_Init(argc, argv);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
#else
  // Non-MPI build: single process
  rank_ = 0;
  size_ = 1;
#endif
}

void MPI::finalize() {
#ifdef SPECFEM_ENABLE_MPI
  int finalized;
  MPI_Finalized(&finalized);

  // Only finalize if not already finalized externally
  if (!finalized) {
    MPI_Finalize();
  }
#endif

  // Reset to uninitialized state
  rank_ = -1;
  size_ = -1;
}

} // namespace specfem
