#include "IO/fortranio/fortran_io.hpp"
#include "IO/fortranio/fortran_io.tpp"
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <string>

void specfem::IO::fortran_IO(std::ifstream &stream, int &buffer_length) {
  if (buffer_length != 0)
    throw std::runtime_error("Error reading fortran file");

  return;
}

void specfem::IO::fortran_read_value(bool *value, std::ifstream &stream,
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

void specfem::IO::fortran_read_value(int *value, std::ifstream &stream,
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

void specfem::IO::fortran_read_value(float *value, std::ifstream &stream,
                                     int &buffer_length) {

  float *temp;
  buffer_length -= ffloat;
  char *ivalue = new char[ffloat];
  if (buffer_length < 0) {
    std::cout << "buffer_length: " << buffer_length << std::endl;
    throw std::runtime_error("Error reading fortran file");
  }
  stream.read(ivalue, ffloat);
  temp = reinterpret_cast<float *>(ivalue);
  *value = *temp;
  delete[] ivalue;
  return;
}

void specfem::IO::fortran_read_value(double *value, std::ifstream &stream,
                                     int &buffer_length) {

  double *temp;
  buffer_length -= fdouble;
  char *ivalue = new char[fdouble];
  if (buffer_length < 0) {
    std::cout << "buffer_length: " << buffer_length << std::endl;
    throw std::runtime_error("Error reading fortran file");
  }
  stream.read(ivalue, fdouble);
  temp = reinterpret_cast<double *>(ivalue);
  *value = *temp;
  delete[] ivalue;
  return;
}

void specfem::IO::fortran_read_value(std::string *value, std::ifstream &stream,
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

template <>
void specfem::IO::fortran_read_value(std::vector<bool> *value,
                                     std::ifstream &stream,
                                     int &buffer_length) {
  int nsize = value->size();
  std::vector<bool> &rvalue = *value;
  for (int i = 0; i < nsize; i++) {
    // Create a temporary bool variable to hold the value
    bool temp_bool;
    specfem::IO::fortran_read_value(&temp_bool, stream, buffer_length);
    // Assign the temporary value to the vector element
    rvalue[i] = temp_bool;
  }
  return;
}
