#ifndef _FORTRAN_IO_TPP
#define _FORTRAN_IO_TPP

#include "IO/fortranio/fortran_io.hpp"
#include "specfem_setup.hpp"
#include <fstream>
#include <iostream>

namespace specfem {
namespace IO {

template <typename T>
void fortran_read_value(std::vector<T> *value,
                                             std::ifstream &stream,
                                             int &buffer_length) {
  int nsize = value->size();
  std::vector<T> &rvalue = *value;

  for (int i = 0; i < nsize; i++) {
    specfem::IO::fortran_read_value(&rvalue[i], stream, buffer_length);
  }

  return;
}


// Template specialization for std::vector<bool>
template <>
void fortran_read_value(std::vector<bool> *value,
                        std::ifstream &stream,
                        int &buffer_length);

template <typename T, typename... Args>
void fortran_IO(std::ifstream &stream, int &buffer_length, T *value,
                Args... values) {

  specfem::IO::fortran_read_value(value, stream, buffer_length);
  specfem::IO::fortran_IO(stream, buffer_length, values...);
  return;
}

template <typename... Args>
void fortran_read_line(std::ifstream &stream, Args... values) {
  int buffer_length;

  if (!stream.is_open()) {
    throw std::runtime_error("Could not find fortran file to read");
  }

  stream.read(reinterpret_cast<char *>(&buffer_length), fint);

  try {
    specfem::IO::fortran_IO(stream, buffer_length, values...);
  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message << "Error reading fortran line with buffer length: "
                  << buffer_length << "\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  stream.read(reinterpret_cast<char *>(&buffer_length), fint);
  return;
}

} // namespace fortran_IO
} // namespace specfem

#endif
