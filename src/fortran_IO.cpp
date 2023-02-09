#include "../include/fortran_IO.h"
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <string>

void specfem::fortran_IO::fortran_IO(std::ifstream &stream,
                                     int &buffer_length) {
  if (buffer_length != 0)
    throw std::runtime_error("Error reading fortran file");

  return;
}

void specfem::fortran_IO::fortran_read_value(bool *value, std::ifstream &stream,
                                             int &buffer_length) {

  buffer_length -= fbool;
  char *ivalue = new char[fbool];
  if (buffer_length < 0) {
    throw std::runtime_error("Error reading fortran file");
  }
  stream.read(ivalue, fbool);

  *value = *reinterpret_cast<bool *>(ivalue);

  delete[] ivalue;
  return;
}

void specfem::fortran_IO::fortran_read_value(int *value, std::ifstream &stream,
                                             int &buffer_length) {

  buffer_length -= fint;
  char *ivalue = new char[fint];
  if (buffer_length < 0) {
    throw std::runtime_error("Error reading fortran file");
  }
  stream.read(ivalue, fint);
  *value = *reinterpret_cast<int *>(ivalue);
  delete[] ivalue;
  return;
}

void specfem::fortran_IO::fortran_read_value(type_real *value,
                                             std::ifstream &stream,
                                             int &buffer_length) {

  double *temp;
  buffer_length -= fdouble;
  char *ivalue = new char[fdouble];
  if (buffer_length < 0) {
    throw std::runtime_error("Error reading fortran file");
  }
  stream.read(ivalue, fdouble);
  temp = reinterpret_cast<double *>(ivalue);
  *value = static_cast<type_real>(*temp);
  delete[] ivalue;
  return;
}

void specfem::fortran_IO::fortran_read_value(std::string *value,
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
