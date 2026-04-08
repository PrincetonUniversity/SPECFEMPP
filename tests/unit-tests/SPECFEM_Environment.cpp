#include "SPECFEM_Environment.hpp"
#include "specfem/mpi.hpp"
#include "specfem/program/context.hpp"
#include <memory>
#include <sstream>

std::shared_ptr<specfem::program::Context> SPECFEMEnvironment::context_ =
    nullptr;
size_t SPECFEMEnvironment::required_mpi_size_ = 1;
bool SPECFEMEnvironment::mpi_size_valid_ = true;
std::string SPECFEMEnvironment::mpi_size_error_ = "";

SPECFEMEnvironment::SPECFEMEnvironment(const size_t nprocs) {
  required_mpi_size_ = nprocs;
}

void SPECFEMEnvironment::SetUp() {
  char **argv = nullptr;
  int argc = 0;
  context_ = std::make_shared<specfem::program::Context>(argc, argv,
                                                         required_mpi_size_);

  // Excluded ranks (those beyond the requested nprocs) must skip
  // all further MPI calls to avoid triggering check_context() exit.
  if (!specfem::MPI::is_active()) {
    mpi_size_valid_ = false;
    mpi_size_error_ = "This rank is excluded from the active communicator";
    return;
  }

  // Check if actual MPI size meets the required size
  const int actual_mpi_size = specfem::MPI::get_size();
  if (actual_mpi_size < static_cast<int>(required_mpi_size_)) {
    mpi_size_valid_ = false;
    std::ostringstream msg;
    msg << "MPI size requirement not met. Required: " << required_mpi_size_
        << ", Actual: " << actual_mpi_size;
    mpi_size_error_ = msg.str();
  }
}

void SPECFEMEnvironment::TearDown() { context_.reset(); }
