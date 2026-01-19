#pragma once

#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Run setup defines run configuration for the simulation
 * @note This object is not used in the current version
 *
 */
class run_setup {

public:
  /**
   * @brief Construct a new run setup object
   *
   * @note This object is not used in the current version
   *
   * @param nproc Number of processors used in the simulation
   * @param nruns Number of simulation runs
   */
  run_setup(int nproc, int nruns) : nproc(nproc), nruns(nruns) {};
  /**
   * @brief Construct a new run setup object
   *
   * @param Node YAML node describing the run configuration
   */
  run_setup(const YAML::Node &Node);

private:
  int nproc; ///< number of processors used in the simulation
  int nruns; ///< Number of simulation runs
};

} // namespace runtime_configuration
} // namespace specfem
