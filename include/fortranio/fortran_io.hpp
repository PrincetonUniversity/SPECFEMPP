#ifndef _FORTRAN_IO_HPP
#define _FORTRAN_IO_HPP

#include "specfem_setup.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace specfem {
namespace fortran_IO {

void fortran_IO(std::ifstream &stream, int &buffer_length);
void fortran_read_value(bool *value, std::ifstream &stream, int &buffer_length);
void fortran_read_value(std::string *value, std::ifstream &stream,
                        int &buffer_length);
void fortran_read_value(type_real *value, std::ifstream &stream,
                        int &buffer_length);
void fortran_read_value(int *value, std::ifstream &stream, int &buffer_length);
template <typename T>
void specfem::fortran_IO::fortran_read_value(std::vector<T> *value,
                                             std::ifstream &stream,
                                             int &buffer_length) {
  int nsize = value->size();
  std::vector<T> &rvalue = *value;

  for (int i = 0; i < nsize; i++) {
    specfem::fortran_IO::fortran_read_value(&rvalue[i], stream, buffer_length);
  }

  return;
}

template <typename T, typename... Args>
void fortran_IO(std::ifstream &stream, int &buffer_length, T *value,
                Args... values) {

  specfem::fortran_IO::fortran_read_value(value, stream, buffer_length);
  specfem::fortran_IO::fortran_IO(stream, buffer_length, values...);
  return;
}
} // namespace fortran_IO
} // namespace specfem

#endif
