#ifndef _FORTRAN_IO_TPP
#define _FORTRAN_IO_TPP

#include "specfem_setup.hpp"
#include <fstream>
#include <iostream>

namespace specfem {
namespace fortran_IO {
template <typename... Args>
void fortran_read_line(std::ifstream &stream, Args... values) {
  int buffer_length;

  if (!stream.is_open()) {
    throw std::runtime_error("Could not find fortran file to read");
  }

  stream.read(reinterpret_cast<char *>(&buffer_length), fint);
  specfem::fortran_IO::fortran_IO(stream, buffer_length, values...);

  stream.read(reinterpret_cast<char *>(&buffer_length), fint);
  return;
}
} // namespace fortran_IO
} // namespace specfem

#endif
