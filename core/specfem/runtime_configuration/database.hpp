#pragma once

#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief database defines the file location of databases
 *
 */
class database {

public:
  /**
   * @brief Construct a new database configuration object
   *
   * @param fortran_database location of fortran database
   */
  database(std::string fortran_database)
      : fortran_database(fortran_database) {};

  /**
   * @brief Construct a new run setup object
   *
   * @param Node YAML node describing the run configuration
   */
  database(const YAML::Node &Node);

  std::string get_databases() const {
    return specfem::MPI::format_proc_filename(this->fortran_database);
  }

private:
  std::string fortran_database; ///< location of fortran binary database
};

} // namespace runtime_configuration
} // namespace specfem
