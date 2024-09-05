#ifndef SPECFEM_READER_WAVEFIELD_HPP
#define SPECFEM_READER_WAVEFIELD_HPP

#include "compute/interface.hpp"
// #include "enumerations/interface.hpp"
#include "reader/reader.hpp"

namespace specfem {
namespace reader {

/**
 * @brief Reader to read wavefield data from disk
 *
 */
template <typename IOLibrary> class wavefield : public reader {

public:
  /**
   * @brief Construct a new reader object
   *
   * @param output_folder Path to output folder or .h5 file
   */
  wavefield(const std::string &output_folder,
            const specfem::compute::assembly &assembly);

  /**
   * @brief Read the wavefield data from disk
   *
   */
  void read() override;

private:
  std::string output_folder; ///< Path to output folder
  specfem::compute::simulation_field<specfem::wavefield::type::buffer>
      buffer; ///< Buffer wavefield to store the data
  specfem::compute::boundary_values boundary_values; ///< Boundary values used
                                                     ///< for backward
                                                     ///< reconstruction during
                                                     ///< adjoint simulations
};

} // namespace reader
} // namespace specfem

#endif /* SPECFEM_READER_WAVEFIELD_HPP */
