#ifndef SPECFEM_READER_WAVEFIELD_HPP
#define SPECFEM_READER_WAVEFIELD_HPP

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "reader/reader.hpp"

namespace specfem {
namespace reader {

/**
 * @brief Base reader class
 *
 */
template <typename IOLibrary> class wavefield : public reader<IOLibrary> {

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
  specfem::compute::impl::field_impl<specfem::enums::element::medium::elastic>
      elastic_field;
  specfem::compute::impl::field_impl<specfem::enums::element::medium::acoustic>
      acoustic_field;
};

} // namespace reader
} // namespace specfem

#endif /* SPECFEM_READER_WAVEFIELD_HPP */
