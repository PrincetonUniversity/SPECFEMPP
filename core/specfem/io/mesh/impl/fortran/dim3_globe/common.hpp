#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace specfem::io::mesh::impl::fortran::dim3_globe_impl {

/** @brief Oldest thin globe database format this reader accepts. */
inline constexpr int globe_database_version_min = 2;

/** @brief Newest thin globe database format this reader accepts. */
inline constexpr int globe_database_version_max = 2;

/** @brief Database material mode indicating properties come from the oracle. */
inline constexpr int material_oracle = 1;

/** @brief Globe database medium tag for acoustic elements. */
inline constexpr int medium_acoustic = 1;

/** @brief Globe database medium tag for elastic elements. */
inline constexpr int medium_elastic = 2;

/**
 * @brief Check that a globe database stream is still readable.
 *
 * @param stream Input stream after a read operation
 * @param section Human-readable database section name for diagnostics
 * @throws std::runtime_error if the stream is in a failed state
 */
void check_stream(const std::ifstream &stream, const std::string &section);

/**
 * @brief Read and validate the leading globe database magic record.
 *
 * The first Fortran record stores a fixed 32-byte magic string followed by the
 * thin-database format version. This helper reads the raw record manually
 * because the payload mixes fixed-length character data and an integer version.
 *
 * @param stream Input stream positioned at the start of the database
 * @return Trimmed magic string and database format version
 * @throws std::runtime_error if record markers or record size are invalid
 */
std::pair<std::string, int> read_magic(std::ifstream &stream);

/**
 * @brief Read a counted integer vector from one Fortran record.
 *
 * The record layout is `count, values(count)`. This is used for globe metadata
 * whose length is stored on disk rather than known by the C++ reader.
 *
 * @param stream Input stream positioned at the counted record
 * @param section Human-readable section name for diagnostics
 * @return Integer values from the record
 * @throws std::runtime_error if the count, marker, or stream state is invalid
 */
std::vector<int> read_counted_ints(std::ifstream &stream,
                                   const std::string &section);

/**
 * @brief Read Fortran logical values encoded as counted integers.
 *
 * @param stream Input stream positioned at the counted logical record
 * @param section Human-readable section name for diagnostics
 * @return Boolean values converted from nonzero integers
 */
std::vector<bool> read_counted_logicals(std::ifstream &stream,
                                        const std::string &section);

/**
 * @brief Read and trim a fixed-length Fortran character record.
 *
 * @param stream Input stream positioned at the character record
 * @param section Human-readable section name for diagnostics
 * @return String with trailing blanks and nulls removed
 * @throws std::runtime_error if record markers or stream state are invalid
 */
std::string read_fixed_string(std::ifstream &stream,
                              const std::string &section);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe_impl
