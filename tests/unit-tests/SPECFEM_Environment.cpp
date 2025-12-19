#include "SPECFEM_Environment.hpp"
#include "specfem/program/context.hpp"
#include <memory>

std::shared_ptr<specfem::program::Context> SPECFEMEnvironment::context_ =
    nullptr;

void SPECFEMEnvironment::SetUp() {
  char **argv = nullptr;
  int argc = 0;
  context_ = std::make_shared<specfem::program::Context>(argc, argv);
}

void SPECFEMEnvironment::TearDown() { context_.reset(); }
