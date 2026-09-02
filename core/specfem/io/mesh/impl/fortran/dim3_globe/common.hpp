#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace specfem::io::mesh::impl::fortran::dim3_globe_impl {

inline constexpr int globe_database_version_min = 2;
inline constexpr int globe_database_version_max = 2;
inline constexpr int material_oracle = 1;
inline constexpr int medium_acoustic = 1;
inline constexpr int medium_elastic = 2;

/** @brief Check that a globe database stream is still readable. */
void check_stream(const std::ifstream &stream, const std::string &section);

/** @brief Read and validate the globe database magic header. */
std::pair<std::string, int> read_magic(std::ifstream &stream);

/** @brief Read a Fortran record containing a count followed by integers. */
std::vector<int> read_counted_ints(std::ifstream &stream,
                                   const std::string &section);

/** @brief Read Fortran logical values encoded as counted integers. */
std::vector<bool> read_counted_logicals(std::ifstream &stream,
                                        const std::string &section);

/** @brief Read a fixed-length Fortran character record. */
std::string read_fixed_string(std::ifstream &stream,
                              const std::string &section);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe_impl
