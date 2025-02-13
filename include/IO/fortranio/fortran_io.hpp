#ifndef _FORTRAN_IO_HPP
#define _FORTRAN_IO_HPP

#include "specfem_setup.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace specfem {
namespace IO {

void fortran_IO(std::ifstream &stream, int &buffer_length);
void fortran_read_value(bool *value, std::ifstream &stream, int &buffer_length);
void fortran_read_value(std::string *value, std::ifstream &stream,
                        int &buffer_length);
void fortran_read_value(float *value, std::ifstream &stream,
                        int &buffer_length);
void fortran_read_value(double *value, std::ifstream &stream,
                        int &buffer_length);
void fortran_read_value(int *value, std::ifstream &stream, int &buffer_length);
} // namespace IO
} // namespace specfem

#endif
