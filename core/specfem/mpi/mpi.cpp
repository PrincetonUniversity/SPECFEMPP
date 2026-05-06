#include "mpi.hpp"

namespace specfem {

// Static storage for MPI rank and size (-1 indicates uninitialized)
int MPI::rank_ = -1;
int MPI::size_ = -1;
MPI_Comm MPI::comm_ = MPI_COMM_WORLD;
specfem::message_tag MPI::message_tag_generator = specfem::message_tag(32768);

void MPI::initialize(int *argc, char ***argv) {
#ifdef SPECFEM_ENABLE_MPI
  int initialized;
  MPI_Initialized(&initialized);

  if (!initialized) {
    MPI_Init(argc, argv);
  }

  comm_ = MPI_COMM_WORLD;

  MPI_Comm_size(comm_, &size_);
  MPI_Comm_rank(comm_, &rank_);
#else
  // Non-MPI build: single process
  rank_ = 0;
  size_ = 1;
  comm_ = MPI_COMM_WORLD; // Dummy value for non-MPI build
#endif
}

void MPI::initialize(int *argc, char ***argv, int nprocs) {
#ifdef SPECFEM_ENABLE_MPI
  int initialized;
  MPI_Initialized(&initialized);

  if (!initialized) {
    MPI_Init(argc, argv);
    // Check that requested nprocs does not exceed world size
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (nprocs > world_size) {
      std::cerr << "Error: Requested nprocs (" << nprocs
                << ") exceeds available world size (" << world_size << ")."
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Get rank from world communicator before using in split
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Only use the first nprocs ranks in the new communicator
    int color = (world_rank < nprocs) ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm_);
  }

  // Handle ranks excluded by subset communicator
  if (comm_ == MPI_COMM_NULL) {
    // Rank is not part of the subset communicator
    rank_ = -1;
    size_ = -1;
    return;
  }

  MPI_Comm_size(comm_, &size_);
  MPI_Comm_rank(comm_, &rank_);
#else
  // Non-MPI build: single process
  rank_ = 0;
  size_ = 1;
  comm_ = MPI_COMM_WORLD; // Dummy value for non-MPI build
#endif
}

void MPI::finalize() {
#ifdef SPECFEM_ENABLE_MPI
  int finalized;
  MPI_Finalized(&finalized);

  // Only finalize if not already finalized externally
  if (!finalized) {
    // Free custom communicator if it was created and is valid
    if (comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&comm_);
      comm_ = MPI_COMM_NULL;
    }
    MPI_Finalize();
  }
#endif

  // Reset to uninitialized state
  rank_ = -1;
  size_ = -1;
}

} // namespace specfem
