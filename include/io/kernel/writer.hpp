#pragma once

#include "enumerations/interface.hpp"
#include "io/writer.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace io {
/**
 * @brief Writer to output misfit kernel data to disk
 *
 * @tparam OutputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename OutputLibrary> class kernel_writer : public writer {
public:
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a writer object
   *
   * @param assembly SPECFEM++ assembly
   * @param output_folder Path to output location (will be an .h5 file if using
   * HDF5, and a folder if using ASCII)
   */
  kernel_writer(const std::string &output_folder);

  /**
   * @brief write the kernel data to disk
   *
   * @param assembly 2D Assembly object
   */
  void write(specfem::assembly::assembly<specfem::dimension::type::dim2>
                 &assembly) override;

  /**
   * @brief write the kernel data to disk
   *
   * @param assembly 3D Assembly object
   */
  void write(specfem::assembly::assembly<specfem::dimension::type::dim3>
                 &assembly) override {
    throw std::runtime_error("3D kernel output not implemented yet");
  }

private:
  std::string output_folder; ///< Path to output folder
};
} // namespace io
} // namespace specfem
