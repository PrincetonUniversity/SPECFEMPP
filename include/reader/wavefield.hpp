#ifndef SPECFEM_READER_WAVEFIELD_HPP
#define SPECFEM_READER_WAVEFIELD_HPP

#include "compute/interface.hpp"
// #include "enumerations/interface.hpp"
#include "reader/reader.hpp"

namespace specfem {
namespace reader {

/**
 * @brief Base reader class
 *
 */
template <typename IOLibrary> class wavefield : public reader {

public:
  /**
   * @brief Construct a new wavefield object
   *
   * @param output_folder Path to output folder
   */
  wavefield(const std::string &output_folder,
            const specfem::compute::assembly &assembly);

  /**
   * @brief Method to execute the read operation
   *
   */
  void read() override;

private:
  std::string output_folder; ///< Path to output folder
  specfem::compute::simulation_field<specfem::wavefield::type::buffer> buffer;
  specfem::compute::boundary_values boundary_values;
};

} // namespace reader
} // namespace specfem

#endif /* SPECFEM_READER_WAVEFIELD_HPP */
