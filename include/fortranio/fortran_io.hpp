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
} // namespace fortran_IO
} // namespace specfem

#endif
