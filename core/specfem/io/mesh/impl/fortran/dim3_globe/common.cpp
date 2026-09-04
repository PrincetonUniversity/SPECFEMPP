#include "specfem/io/mesh/impl/fortran/dim3_globe/common.hpp"

#include <array>
#include <stdexcept>

namespace specfem::io::mesh::impl::fortran::dim3_globe_impl {

void check_stream(const std::ifstream &stream, const std::string &section) {
  if (!stream) {
    throw std::runtime_error("Failed to read globe mesh database section: " +
                             section);
  }
}

std::pair<std::string, int> read_magic(std::ifstream &stream) {
  int record_size = 0;
  int trailing_size = 0;
  int version = 0;
  std::array<char, 32> magic{};

  stream.read(reinterpret_cast<char *>(&record_size), sizeof(record_size));
  if (record_size != static_cast<int>(magic.size() + sizeof(version))) {
    throw std::runtime_error("Invalid SPECFEM++ globe database header size");
  }
  stream.read(magic.data(), magic.size());
  stream.read(reinterpret_cast<char *>(&version), sizeof(version));
  stream.read(reinterpret_cast<char *>(&trailing_size), sizeof(trailing_size));
  check_stream(stream, "header");
  if (trailing_size != record_size) {
    throw std::runtime_error("Mismatched Fortran record markers in globe "
                             "database header");
  }

  std::string result(magic.data(), magic.size());
  const auto last = result.find_last_not_of(' ');
  result.resize(last == std::string::npos ? 0 : last + 1);
  return { result, version };
}

std::vector<int> read_counted_ints(std::ifstream &stream,
                                   const std::string &section) {
  int record_size = 0;
  int trailing_size = 0;
  int count = 0;
  stream.read(reinterpret_cast<char *>(&record_size), sizeof(record_size));
  stream.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (count < 0 || record_size != static_cast<int>((count + 1) * sizeof(int))) {
    throw std::runtime_error("Invalid " + section +
                             " record in globe mesh database");
  }
  std::vector<int> values(count);
  if (count > 0) {
    stream.read(reinterpret_cast<char *>(values.data()), count * sizeof(int));
  }
  stream.read(reinterpret_cast<char *>(&trailing_size), sizeof(trailing_size));
  check_stream(stream, section);
  if (trailing_size != record_size) {
    throw std::runtime_error("Mismatched Fortran record markers in " + section);
  }
  return values;
}

std::vector<bool> read_counted_logicals(std::ifstream &stream,
                                        const std::string &section) {
  const auto raw = read_counted_ints(stream, section);
  std::vector<bool> values(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    values[i] = raw[i] != 0;
  }
  return values;
}

std::string read_fixed_string(std::ifstream &stream,
                              const std::string &section) {
  int record_size = 0;
  int trailing_size = 0;
  stream.read(reinterpret_cast<char *>(&record_size), sizeof(record_size));
  if (record_size < 0) {
    throw std::runtime_error("Invalid " + section +
                             " record in globe mesh database");
  }
  std::string value(static_cast<std::size_t>(record_size), ' ');
  if (record_size > 0) {
    stream.read(value.data(), record_size);
  }
  stream.read(reinterpret_cast<char *>(&trailing_size), sizeof(trailing_size));
  check_stream(stream, section);
  if (trailing_size != record_size) {
    throw std::runtime_error("Mismatched Fortran record markers in " + section);
  }
  const auto last = value.find_last_not_of(" \0", std::string::npos, 2);
  value.resize(last == std::string::npos ? 0 : last + 1);
  return value;
}

} // namespace specfem::io::mesh::impl::fortran::dim3_globe_impl
