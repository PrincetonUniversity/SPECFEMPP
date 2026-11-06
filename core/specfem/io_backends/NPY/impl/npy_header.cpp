#include <algorithm>
#include <complex>
#include <cstdint>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "specfem/io_backends/NPY/impl/npy_header.hpp"

namespace specfem::io_backends::impl::NPY {

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
  // Byte strings have no byte order; NumPy marks them '|' (not applicable).
  dict += (type_char == 'S') ? '|' : []() {
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

NPYHeader read_npy_header(std::ifstream &file, const char type_char) {
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

  return { shape, byte_size, fortran_order };
}

NPYHeader parse_npy_header(std::ifstream &file, const char type_char,
                           const size_t type_size) {
  const NPYHeader header = read_npy_header(file, type_char);
  if (header.item_size != type_size) {
    throw std::runtime_error("parse_npy_header: byte size mismatch");
  }
  return header;
}

void check_fortran_order(const NPYHeader &header, const bool fortran_order,
                         const std::string &path) {
  if (header.shape.size() > 1 && header.fortran_order != fortran_order) {

    const auto order_name = [](const bool is_fortran) {
      return is_fortran ? "Fortran (column-major)" : "C (row-major)";
    };
    throw std::runtime_error("ERROR : " + path + " is stored in " +
                             order_name(header.fortran_order) +
                             " order, but the view it is read into is " +
                             order_name(fortran_order));
  }
}

size_t string_record_width(const std::string *data, const size_t count) {
  size_t width = 1;
  for (size_t i = 0; i < count; ++i) {
    width = std::max(width, data[i].size());
  }
  return width;
}

std::vector<char> pack_string_records(const std::string *data,
                                      const size_t count, const size_t width) {
  std::vector<char> records(count * width, '\0');
  for (size_t i = 0; i < count; ++i) {
    if (data[i].size() > width) {
      throw std::runtime_error(
          "pack_string_records: string longer than record width");
    }
    std::copy(data[i].begin(), data[i].end(), records.begin() + i * width);
  }
  return records;
}

void unpack_string_records(const std::vector<char> &records, const size_t count,
                           const size_t width, std::string *data) {
  for (size_t i = 0; i < count; ++i) {
    const char *record = records.data() + i * width;
    size_t length = 0;
    while (length < width && record[length] != '\0') {
      ++length;
    }
    data[i].assign(record, length);
  }
}

} // namespace specfem::io_backends::impl::NPY
