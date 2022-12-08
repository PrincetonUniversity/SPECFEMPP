#include "../../include/specfem_mpi.h"
#include <gtest/gtest.h>

class MPIEnvironment : public ::testing::Environment {
public:
  virtual void SetUp();
  virtual void TearDown();
  static specfem::MPI::MPI *mpi_;
};
