#pragma once

#include "specfem/io/writer.hpp"

namespace specfem {
namespace io {
/**
 * @brief Writer for outputting sensitivity kernels to disk
 *
 * Template-based writer for adjoint sensitivity kernel data supporting multiple
 * I/O backends. Kernels represent the gradient of a misfit function with
 * respect to material properties.
 *
 * @tparam OutputLibrary Backend library type (HDF5, ASCII, NPY, NPZ, or ADIOS2)
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
  void write(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                 &assembly) override;

  /**
   * @brief write the kernel data to disk
   *
   * @param assembly 3D Assembly object
   */
  void write(specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
                 &assembly) override {
    throw std::runtime_error("3D kernel output not implemented yet");
  }

private:
  std::string output_folder; ///< Path to output folder
};
} // namespace io
} // namespace specfem
