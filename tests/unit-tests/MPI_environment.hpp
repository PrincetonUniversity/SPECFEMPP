#include "specfem_mpi/interface.hpp"
#include <gtest/gtest.h>

class MPIEnvironment : public ::testing::Environment {
public:
  void SetUp();
  void TearDown();

  static specfem::MPI::MPI *get_mpi() { return mpi_.get(); }

private:
  static std::shared_ptr<specfem::MPI::MPI> mpi_;
};
