#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace io {
/**
 * @brief Base writer class for outputting simulation data
 *
 * Abstract interface for implementing format-specific writers.
 * Derived classes must implement write() for both 2D and 3D assemblies.
 */
class writer {
public:
  /**
   * @brief Method to execute the write operation
   *
   * @param assembly 2D Assembly object
   *
   */
  virtual void write(specfem::assembly::assembly<specfem::dimension::type::dim2>
                         &assembly) = 0;

  /**
   * @brief Method to execute the write operation
   *
   * @param assembly 3D Assembly object
   *
   */
  virtual void write(specfem::assembly::assembly<specfem::dimension::type::dim3>
                         &assembly) = 0;
};

} // namespace io
} // namespace specfem
