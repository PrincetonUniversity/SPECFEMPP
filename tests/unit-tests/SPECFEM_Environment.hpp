#pragma once

#include "specfem/program/context.hpp"
#include "specfem_mpi/interface.hpp"
#include <gtest/gtest.h>
#include <memory>

class SPECFEMEnvironment : public ::testing::Environment {
public:
  void SetUp() override;
  void TearDown() override;

  static specfem::MPI::MPI *get_mpi() {
    return context_ ? context_->get_mpi() : nullptr;
  }

private:
  static std::shared_ptr<specfem::program::Context> context_;
};
