#include "../include/fortran_IO.h"
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <string>

void IO::fortran_IO::fortran_IO(std::ifstream &stream, int &buffer_length) {
  if (buffer_length != 0)
    throw std::runtime_error("Error reading fortran file");

  return;
}

void IO::fortran_IO::fortran_read_value(bool *value, std::ifstream &stream,
                                        int &buffer_length) {

  buffer_length -= fbool;
  if (buffer_length < 0) {
    throw std::runtime_error("Error reading fortran file");
  }
  stream.read(reinterpret_cast<char *>(value), fbool);
  return;
}

void IO::fortran_IO::fortran_read_value(int *value, std::ifstream &stream,
                                        int &buffer_length) {

  buffer_length -= fint;
  if (buffer_length < 0) {
    throw std::runtime_error("Error reading fortran file");
  }
  stream.read(reinterpret_cast<char *>(value), fint);
  return;
}

void IO::fortran_IO::fortran_read_value(type_real *value, std::ifstream &stream,
                                        int &buffer_length) {

  double temp;
  buffer_length -= fdouble;
  if (buffer_length < 0) {
    throw std::runtime_error("Error reading fortran file");
  }
  stream.read(reinterpret_cast<char *>(&temp), fdouble);
  *value = static_cast<type_real>(temp);
  return;
}

void IO::fortran_IO::fortran_read_value(std::string *value,
                                        std::ifstream &stream,
                                        int &buffer_length) {
  // reading a string has few errors. There seem to unknown characters at the
  // end of the string
  char temp[fchar];
  value->clear();
  buffer_length -= fchar;
  if (buffer_length < 0) {
    throw std::runtime_error("Error reading fortran file");
  }
  stream.read(reinterpret_cast<char *>(&temp), fchar);
  value->append(temp);
  boost::algorithm::trim(*value);
  return;
}
