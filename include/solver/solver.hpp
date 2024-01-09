#ifndef _SOLVER_HPP
#define _SOLVER_HPP

#include "domain/interface.hpp"
#include "timescheme/interface.hpp"

namespace specfem {
namespace solver {

/**
 * @brief Base solver class
 *
 * Solver class is the base class for all solver algorithms (implicit or
 * explicit). The class contains a pure virtual function run() that must be
 * implemented by the derived solver implementations.
 *
 */
class solver {

public:
  /**
   * @brief Run solver algorithm
   *
   */
  virtual void run() = 0;

  virtual ~solver() = default;
};

} // namespace solver
} // namespace specfem

#endif
