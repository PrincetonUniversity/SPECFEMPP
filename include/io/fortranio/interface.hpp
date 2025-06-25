#ifndef _FORTRAN_IO_INTERFACE_HPP
#define _FORTRAN_IO_INTERFACE_HPP

#include "fortran_io.hpp"
#include "fortran_io.tpp"
#include <fstream>

namespace specfem {
namespace io {
/**
 * @brief Read a line from fortran unformatted binary file
 *
 * This function is the core of the fortran binary file reading. It reads a line
 * from the file and assigns the values to the variables passed as arguments.
 *
 * @tparam Args Argument can be of the type `bool`, `int`, `type_real`,
 *              `string`, or `vector<T = bool, int, type_real, string>`
 * @param stream An open file stream.
 * @param values Comma separated list of variable addresses to be read.
 * @throws std::runtime_error if an error occurs while reading the line
 *
 * @code{.cpp} // Example of how to use this function
 * int value1, value2;
 * specfem::io::fortran_read_line(stream, &value1, &value2);
 *
 * // Example of how to use this function with a vector
 * std::vector<int> values(10);
 * specfem::io::fortran_read_line(stream, &values);
 * @endcode
 */
template <typename... Args>
void fortran_read_line(std::ifstream &stream, Args... values);
} // namespace io
} // namespace specfem

#endif
