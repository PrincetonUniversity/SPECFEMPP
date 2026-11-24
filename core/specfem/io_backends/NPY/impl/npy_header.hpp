#pragma once

#include <complex>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "map_type.tpp"

namespace specfem::io_backends::impl::NPY {
/**
 * @brief A class wrapping a vector of char to facilitate building npy headers.
 */
class NPYString : public std::vector<char> {
public:
  using std::vector<char>::vector;
  template <typename T> NPYString &operator+=(const T rhs);
};

template <> NPYString &NPYString::operator+=(const std::string rhs);

template <> NPYString &NPYString::operator+=(const char *rhs);

template <typename T> NPYString &NPYString::operator+=(const T rhs) {
  // write in little endian
  for (size_t byte = 0; byte < sizeof(T); byte++) {
    char val = *((char *)&rhs + byte);
    this->push_back(val);
  }
  return *this;
}

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
NPYString create_npy_header(const std::vector<size_t> &shape,
                            const char type_char, const size_t type_size,
                            bool fortran_order);

template <typename value_type>
NPYString create_npy_header(const std::vector<size_t> &shape,
                            bool fortran_order = true) {
  return create_npy_header(shape, map_type<value_type>(), sizeof(value_type),
                           fortran_order);
}

/**
 * @brief Array metadata stored in a NumPy .npy file header
 */
struct NPYHeader {
  std::vector<size_t> shape; ///< Dimensions of the array
  size_t item_size = 0;      ///< Size of each element in bytes
};

/**
 * @brief Read a NumPy .npy file header without constraining the element size
 *
 * Used for variable-width types such as byte strings (`|S<n>`), whose element
 * size is only known from the file.
 *
 * @param file File stream to the .npy file, positioned at the start of the
 * header
 * @param type_char Expected NumPy type character
 * @return NPYHeader Shape and element size stored in the header
 * @throws std::runtime_error If the header cannot be parsed, is not in
 * Fortran order, is big-endian, or its type does not match `type_char`
 *
 * @note Modified from cnpy library (MIT License)
 * https://github.com/rogersce/cnpy
 */
NPYHeader read_npy_header(std::ifstream &file, const char type_char);

/**
 * @brief Parse a NumPy .npy file header
 *
 * @param file File stream to the .npy file, positioned at the start of the
 * header
 * @param type_char Expected NumPy type character
 * @param type_size Expected size of each element in bytes
 * @return std::vector<size_t> Dimensions of the array
 * @throws std::runtime_error If the header cannot be parsed or does not match
 * the expected type and element size
 */
std::vector<size_t> parse_npy_header(std::ifstream &file, const char type_char,
                                     const size_t type_size);

template <typename value_type>
std::vector<size_t> parse_npy_header(std::ifstream &file) {
  return parse_npy_header(file, map_type<value_type>(), sizeof(value_type));
}

/**
 * @brief Record width used to store strings as a NumPy `|S<n>` array
 *
 * Matches NumPy, which sizes byte-string arrays to their longest element.
 *
 * @param data Strings to store
 * @param count Number of strings
 * @return size_t Length of the longest string, at least 1
 */
size_t string_record_width(const std::string *data, const size_t count);

/**
 * @brief Pack strings into null-padded fixed-width records
 *
 * @param data Strings to pack
 * @param count Number of strings
 * @param width Record width in bytes, at least the longest string length
 * @return std::vector<char> `count * width` bytes, one record per string
 */
std::vector<char> pack_string_records(const std::string *data,
                                      const size_t count, const size_t width);

/**
 * @brief Unpack null-padded fixed-width records into strings
 *
 * @param records `count * width` bytes, one record per string
 * @param count Number of strings
 * @param width Record width in bytes
 * @param data Output strings, trailing null padding removed
 */
void unpack_string_records(const std::vector<char> &records, const size_t count,
                           const size_t width, std::string *data);

} // namespace specfem::io_backends::impl::NPY
