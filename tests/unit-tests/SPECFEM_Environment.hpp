#pragma once

#include "specfem/program/context.hpp"

#include <gtest/gtest.h>
#include <memory>

class SPECFEMEnvironment : public ::testing::Environment {
public:
  void SetUp() override;
  void TearDown() override;

private:
  static std::shared_ptr<specfem::program::Context> context_;
};
