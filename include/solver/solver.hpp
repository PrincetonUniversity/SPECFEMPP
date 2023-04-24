#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "domain.h"
#include "timescheme/interface.hpp"

namespace specfem {
namespace solver {

/**
 * @brief Base solver class
 *
 */
class solver {

public:
  /**
   * @brief Run solver algorithm
   *
   */
  virtual void run(){};
};

} // namespace solver
} // namespace specfem

#endif
