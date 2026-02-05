#ifndef _FORTRAN_IO_HPP
#define _FORTRAN_IO_HPP

#include "specfem_setup.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace specfem {
namespace io {

/**
 * @brief Process Fortran record markers
 *
 * @param stream Input file stream
 * @param buffer_length Reference to store record length
 */
void fortran_IO(std::ifstream &stream, int &buffer_length);

/**
 * @brief Read boolean value from Fortran binary stream
 */
void fortran_read_value(bool *value, std::ifstream &stream, int &buffer_length);

/**
 * @brief Read string value from Fortran binary stream
 */
void fortran_read_value(std::string *value, std::ifstream &stream,
                        int &buffer_length);

/**
 * @brief Read float value from Fortran binary stream
 */
void fortran_read_value(float *value, std::ifstream &stream,
                        int &buffer_length);

/**
 * @brief Read double value from Fortran binary stream
 */
void fortran_read_value(double *value, std::ifstream &stream,
                        int &buffer_length);

/**
 * @brief Read integer value from Fortran binary stream
 */
void fortran_read_value(int *value, std::ifstream &stream, int &buffer_length);
} // namespace io
} // namespace specfem

#endif
