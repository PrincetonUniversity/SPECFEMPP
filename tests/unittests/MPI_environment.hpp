#include "specfem_mpi/interface.hpp"
#include <gtest/gtest.h>

class MPIEnvironment : public ::testing::Environment {
public:
  virtual void SetUp();
  virtual void TearDown();
  static specfem::MPI::MPI *mpi_;
};
