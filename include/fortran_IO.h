#ifndef FORTRAN_IO_H
#define FORTRAN_IO_H

#include "../include/config.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace IO::fortran_IO {

void fortran_IO(std::ifstream &stream, int &buffer_length);
void fortran_read_value(bool *value, std::ifstream &stream, int &buffer_length);
void fortran_read_value(std::string *value, std::ifstream &stream,
                        int &buffer_length);
void fortran_read_value(type_real *value, std::ifstream &stream,
                        int &buffer_length);
void fortran_read_value(int *value, std::ifstream &stream, int &buffer_length);

template <typename T, typename... Args>
void fortran_IO(std::ifstream &stream, int &buffer_length, T *value,
                Args... values) {

  IO::fortran_IO::fortran_read_value(value, stream, buffer_length);
  IO::fortran_IO::fortran_IO(stream, buffer_length, values...);
  return;
}

template <typename... Args>
void fortran_read_line(std::ifstream &stream, Args... values) {
  int buffer_length;

  stream.read(reinterpret_cast<char *>(&buffer_length), fint);
  IO::fortran_IO::fortran_IO(stream, buffer_length, values...);

  stream.read(reinterpret_cast<char *>(&buffer_length), fint);
  return;
}

template <typename T>
void IO::fortran_IO::fortran_read_value(std::vector<T> *value,
                                        std::ifstream &stream,
                                        int &buffer_length) {
  int nsize = value->size();
  std::vector<T> &rvalue = *value;

  for (int i = 0; i < nsize; i++) {
    IO::fortran_IO::fortran_read_value(&rvalue[i], stream, buffer_length);
  }

  return;
}
} // namespace IO::fortran_IO

#endif
