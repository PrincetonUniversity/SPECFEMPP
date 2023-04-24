#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "../../include/domain.h"
#include "../../include/timescheme.h"

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
