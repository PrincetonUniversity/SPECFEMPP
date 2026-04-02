#pragma once

#include "specfem/program/context.hpp"

#include <gtest/gtest.h>
#include <memory>

class SPECFEMEnvironment : public ::testing::Environment {
public:
  SPECFEMEnvironment() = default;
  ~SPECFEMEnvironment() override = default;

  SPECFEMEnvironment(const size_t nnodes);

  void SetUp() override;
  void TearDown() override;

  static bool IsMPISizeValid() { return mpi_size_valid_; }
  static std::string GetMPISizeError() { return mpi_size_error_; }

private:
  static std::shared_ptr<specfem::program::Context> context_;
  static size_t required_mpi_size_;
  static bool mpi_size_valid_;
  static std::string mpi_size_error_;
};
