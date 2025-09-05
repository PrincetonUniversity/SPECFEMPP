#pragma once

#include <complex>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "map_type.tpp"

namespace specfem::io::impl::NPY {
/**
 * @brief Create a NumPy .npy file header
 *
 * This function generates the header for a NumPy .npy file based on the
 * provided shape, data type, and array ordering information. The header follows
 * the NumPy file format specification.
 *
 * @tparam value_type The data type of the array elements
 * @param shape Vector containing the dimensions of the array
 * @param fortran_order Boolean indicating whether the array uses Fortran-style
 * ordering
 *
 * @return std::string The generated header as a vector of characters
 *
 * @note Modified from cnpy library (MIT License)
 * https://github.com/rogersce/cnpy
 *
 * @example
 * For a 3x4 float array using Fortran ordering on a little-endian system,
 * the header dictionary might look like:
 * {'descr': '<f4', 'fortran_order': True, 'shape': (3, 4), }
 */
std::string impl_create_npy_header(const std::vector<size_t> &shape,
                                   const char type_char, const size_t type_size,
                                   bool fortran_order);

template <typename value_type>
std::string create_npy_header(const std::vector<size_t> &shape,
                              bool fortran_order = true) {
  return impl_create_npy_header(shape, map_type<value_type>(),
                                sizeof(value_type), fortran_order);
}

/**
 * @brief Parse a NumPy .npy file header
 *
 * This function extracts metadata from a NumPy .npy file header, including word
 * size, array shape, and array ordering information.
 *
 * @param file File stream to the .npy file, positioned at the start of the
 * header
 * @param byte_size Output parameter that will contain the size of each element
 * in bytes
 * @param shape Output vector that will contain the dimensions of the array
 * @param fortran_order Output parameter that will indicate whether the array
 * uses Fortran-style ordering
 * @throws std::runtime_error If the header cannot be properly parsed
 *
 * @note Modified from cnpy library (MIT License)
 * https://github.com/rogersce/cnpy
 */
std::vector<size_t> impl_parse_npy_header(std::ifstream &file,
                                          const char type_char,
                                          const size_t type_size,
                                          bool fortran_order);

template <typename value_type>
std::vector<size_t> parse_npy_header(std::ifstream &file,
                                     bool fortran_order = true) {
  return impl_parse_npy_header(file, map_type<value_type>(), sizeof(value_type),
                               fortran_order);
}

} // namespace specfem::io::impl::NPY
