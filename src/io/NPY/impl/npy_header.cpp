#include <complex>
#include <cstdint>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/NPY/impl/npy_header.hpp"

namespace specfem::io::impl::NPY {

template <> NPYString &NPYString::operator+=(const std::string rhs) {
  this->insert(this->end(), rhs.begin(), rhs.end());
  return *this;
}

template <> NPYString &NPYString::operator+=(const char *rhs) {
  // write in little endian
  size_t len = strlen(rhs);
  this->reserve(len);
  for (size_t byte = 0; byte < len; byte++) {
    this->push_back(rhs[byte]);
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
                            bool fortran_order) {
  NPYString dict;
  dict += "{'descr': '";
  dict += []() {
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
  }();
  dict += type_char;
  dict += std::to_string(type_size);
  dict += "', 'fortran_order': ";
  dict += (fortran_order ? "True" : "False");
  dict += ", 'shape': (";
  dict += std::to_string(shape[0]);
  for (size_t i = 1; i < shape.size(); i++) {
    dict += ", ";
    dict += std::to_string(shape[i]);
  }
  if (shape.size() == 1)
    dict += ",";
  dict += "), }";
  // pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10
  // bytes. dict needs to end with \n
  int remainder = 16 - (10 + dict.size()) % 16;
  dict.insert(dict.end(), remainder, ' ');
  dict.back() = '\n';

  NPYString header;
  header += (char)0x93;
  header += "NUMPY";
  header += (char)0x01; // major version of numpy format
  header += (char)0x00; // minor version of numpy format
  header += (uint16_t)dict.size();
  header.insert(header.end(), dict.begin(), dict.end());

  return header;
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
std::vector<size_t> parse_npy_header(std::ifstream &file, const char type_char,
                                     const size_t type_size) {
  char buffer[11];
  file.read(buffer, 11 * sizeof(char));
  // Read the header into a string
  std::string header;
  std::getline(file, header);

  size_t loc1, loc2;

  // fortran order
  loc1 = header.find("fortran_order");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'fortran_order'");
  loc1 += 16;
  bool fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

  if (!fortran_order) {
    throw std::runtime_error(
        "parse_npy_header: only fortran_order=True is supported");
  }

  // shape
  loc1 = header.find("(");
  loc2 = header.find(")");
  if (loc1 == std::string::npos || loc2 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: '(' or ')'");

  std::regex num_regex("[0-9][0-9]*");
  std::smatch sm;
  std::vector<size_t> shape;

  std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
  while (std::regex_search(str_shape, sm, num_regex)) {
    shape.push_back(std::stoul(sm[0].str()));
    str_shape = sm.suffix().str();
  }

  // endian, word size, data type
  // byte order code | stands for not applicable.
  // not sure when this applies except for byte array
  loc1 = header.find("descr");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'descr'");
  loc1 += 9;
  bool littleEndian =
      (header[loc1] == '<' || header[loc1] == '|' ? true : false);
  if (!littleEndian) {
    throw std::runtime_error(
        "parse_npy_header: only little-endian format is supported");
  }
  if (header[loc1 + 1] != type_char) {
    throw std::runtime_error("parse_npy_header: type mismatch");
  }
  std::string str_ws = header.substr(loc1 + 2);
  loc2 = str_ws.find("'");
  size_t byte_size = atoi(str_ws.substr(0, loc2).c_str());
  if (byte_size != type_size) {
    throw std::runtime_error("parse_npy_header: byte size mismatch");
  }

  return shape;
}

} // namespace specfem::io::impl::NPY
