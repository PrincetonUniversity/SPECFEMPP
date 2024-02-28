#ifndef _FORTRAN_IO_INTERFACE_HPP
#define _FORTRAN_IO_INTERFACE_HPP

#include "fortran_io.hpp"
#include "fortran_io.tpp"
#include <fstream>

namespace specfem {
namespace IO {
/**
 * @brief Read a line from fortran unformatted binary file
 *
 * @tparam Args Argument can be of the type bool, int, type_real, string,
 * vector<T = bool, int, type_real, string>
 * @param stream An open file stream.
 * @param values Comma separated list of variable addresses to be read.
 */
template <typename... Args>
void fortran_read_line(std::ifstream &stream, Args... values);
} // namespace IO
} // namespace specfem

#endif
