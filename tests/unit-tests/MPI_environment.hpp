#pragma once

#include "specfem_mpi/interface.hpp"
#include <gtest/gtest.h>
#include <memory>

class MPIEnvironment : public ::testing::Environment {
public:
  void SetUp();
  void TearDown();

  // Return shared_ptr so tests can use it directly or convert to weak_ptr
  static std::shared_ptr<specfem::MPI::MPI> get_mpi() { return mpi_; }

private:
  static std::shared_ptr<specfem::MPI::MPI> mpi_;
};
