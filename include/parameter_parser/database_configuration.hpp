#ifndef _PARAMETER_RUNTIME_CONFIGURATION_HPP
#define _PARAMETER_RUNTIME_CONFIGURATION_HPP

#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief database_configuration defines the file location of databases
 *
 */
class database_configuration {

public:
  /**
   * @brief Construct a new database configuration object
   *
   * @param fortran_database location of fortran database
   */
  database_configuration(std::string fortran_database)
      : fortran_database(fortran_database) {};

  /**
   * @brief Construct a new run setup object
   *
   * @param Node YAML node describing the run configuration
   */
  database_configuration(const YAML::Node &Node);

  std::string get_databases() const { return this->fortran_database; }

private:
  std::string fortran_database; ///< location of fortran binary database
};

} // namespace runtime_configuration
} // namespace specfem

#endif
