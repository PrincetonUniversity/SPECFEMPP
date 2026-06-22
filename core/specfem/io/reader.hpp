#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"

namespace specfem {
namespace io {
/**
 * @brief Base reader class for loading simulation data
 *
 * Abstract interface for implementing format-specific readers.
 * Derived classes must implement read() for both 2D and 3D assemblies.
 */
class reader {
public:
  /**
   * @brief Method to execute the read operation
   *
   * @param assembly 2D Assembly object
   *
   */
  virtual void
  read(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
           &assembly) = 0;

  /**
   * @brief Method to execute the read operation
   *
   * @param assembly 3D Assembly object
   *
   */
  virtual void
  read(specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
           &assembly) = 0;
};
} // namespace io
} // namespace specfem
