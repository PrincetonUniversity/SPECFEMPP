#pragma once

namespace specfem::io::impl::NPY {
/**
 * @brief Maps C++ types to NumPy type characters
 *
 * This function converts C++ type information to the corresponding NumPy type
 * character:
 * - 'f' for floating point types (float, double, long double)
 * - 'i' for signed integer types (int, char, short, long, long long)
 * - 'u' for unsigned integer types (unsigned char, unsigned short, unsigned
 * long, unsigned long long, unsigned int)
 * - 'b' for boolean
 * - 'c' for complex types (std::complex<float>, std::complex<double>,
 * std::complex<long double>)
 * - '?' for unknown types
 *
 * @param t Reference to a std::type_info object representing the type to map
 * @return char The corresponding NumPy type character
 *
 * @note Modified from cnpy library (MIT License)
 * https://github.com/rogersce/cnpy
 */
template <typename value_type> char map_type();

} // namespace specfem::io::impl::NPY
