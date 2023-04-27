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
   * @param source_database location of source file
   */
  database_configuration(std::string fortran_database,
                         std::string source_database)
      : fortran_database(fortran_database), source_database(source_database){};
  /**
   * @brief Construct a new run setup object
   *
   * @param Node YAML node describing the run configuration
   */
  database_configuration(const YAML::Node &Node);

  std::tuple<std::string, std::string> get_databases() const {
    return std::make_tuple(this->fortran_database, this->source_database);
  }

private:
  std::string fortran_database; ///< location of fortran binary database
  std::string source_database;  ///< location of sources file
};

} // namespace runtime_configuration
} // namespace specfem

#endif
