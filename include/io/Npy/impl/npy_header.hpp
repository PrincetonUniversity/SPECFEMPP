#pragma once

#include <complex>
#include <regex>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace specfem::io::impl::Npy {
/**
 * @brief Maps C++ types to NumPy type characters
 *
 * This function converts C++ type information to the corresponding NumPy type
 * character:
 * - 'f' for floating point types (float, double, long double)
 * - 'i' for signed integer types (int, char, short, long, long long)
 * - 'u' for unsigned integer types (unsigned char, unsigned short, unsigned
 * long, unsigned long long, unsigned int)
 * - 'b' for boolean
 * - 'c' for complex types (std::complex<float>, std::complex<double>,
 * std::complex<long double>)
 * - '?' for unknown types
 *
 * @param t Reference to a std::type_info object representing the type to map
 * @return char The corresponding NumPy type character
 *
 * @note Modified from cnpy library (MIT License)
 * https://github.com/rogersce/cnpy
 */

template <typename value_type> char map_type() {
  if (typeid(value_type) == typeid(float))
    return 'f';
  if (typeid(value_type) == typeid(double))
    return 'f';
  if (typeid(value_type) == typeid(long double))
    return 'f';

  if (typeid(value_type) == typeid(int))
    return 'i';
  if (typeid(value_type) == typeid(char))
    return 'i';
  if (typeid(value_type) == typeid(short))
    return 'i';
  if (typeid(value_type) == typeid(long))
    return 'i';
  if (typeid(value_type) == typeid(long long))
    return 'i';

  if (typeid(value_type) == typeid(unsigned char))
    return 'u';
  if (typeid(value_type) == typeid(unsigned short))
    return 'u';
  if (typeid(value_type) == typeid(unsigned long))
    return 'u';
  if (typeid(value_type) == typeid(unsigned long long))
    return 'u';
  if (typeid(value_type) == typeid(unsigned int))
    return 'u';

  if (typeid(value_type) == typeid(bool))
    return 'b';

  if (typeid(value_type) == typeid(std::complex<float>))
    return 'c';
  if (typeid(value_type) == typeid(std::complex<double>))
    return 'c';
  if (typeid(value_type) == typeid(std::complex<long double>))
    return 'c';

  else
    return '?';
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
 * @return std::vector<char> The generated header as a vector of characters
 *
 * @note Modified from cnpy library (MIT License)
 * https://github.com/rogersce/cnpy
 */
template <typename value_type>
std::vector<char> create_npy_header(const std::vector<size_t> &shape,
                                    bool fortran_order = true) {
  std::string dict;
  dict += "{'descr': '";
  dict += []() {
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
  }();
  dict += map_type<value_type>();
  dict += std::to_string(sizeof(value_type));
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

  std::vector<char> header;
  header.push_back((char)0x93);

  // Add "NUMPY" string
  const char *numpy_str = "NUMPY";
  header.insert(header.end(), numpy_str, numpy_str + 5);

  header.push_back((char)0x01); // major version of numpy format
  header.push_back((char)0x00); // minor version of numpy format

  // Add dict size as uint16_t (2 bytes)
  uint16_t dict_size = dict.size();
  header.push_back(dict_size & 0xFF);
  header.push_back((dict_size >> 8) & 0xFF);

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
template <typename value_type>
std::vector<size_t> parse_npy_header(std::ifstream &file,
                                     bool fortran_order = true) {
  char buffer[256];
  file.read(buffer, 11 * sizeof(char));
  // Read the header into a string
  file.getline(buffer, 256);
  std::string header(buffer);
  if (header.size() == 256) {
    throw std::runtime_error("parse_npy_header: header size out of range");
  }

  size_t loc1, loc2;

  // fortran order
  loc1 = header.find("fortran_order");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'fortran_order'");
  loc1 += 16;
  fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

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
  char type_char = header[loc1 + 1];
  if (type_char != map_type<value_type>()) {
    throw std::runtime_error("parse_npy_header: type mismatch");
  }
  std::string str_ws = header.substr(loc1 + 2);
  loc2 = str_ws.find("'");
  size_t byte_size = atoi(str_ws.substr(0, loc2).c_str());
  if (byte_size != sizeof(value_type)) {
    throw std::runtime_error("parse_npy_header: byte size mismatch");
  }

  return shape;
}

} // namespace specfem::io::impl::Npy
