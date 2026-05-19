#pragma once

#include "specfem/setup.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace specfem {
namespace io {
namespace sources_impl {

/// Strip leading and trailing whitespace.
inline std::string trim(const std::string &s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

/// Lowercase a string (for case-insensitive matching).
inline std::string to_lower(const std::string &s) {
  std::string out = s;
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}

/// Replace Fortran-style 'd'/'D' exponent with 'e'/'E' so std::stod works.
inline std::string fortran_to_cpp_double(const std::string &s) {
  std::string out = s;
  for (auto &c : out) {
    if (c == 'd' || c == 'D')
      c = 'e';
  }
  return out;
}

/// Read all non-blank lines from a file.
inline std::vector<std::string>
read_nonempty_lines(const std::string &file_path) {
  std::ifstream ifs(file_path);
  if (!ifs.is_open())
    throw std::runtime_error("Cannot open source file: " + file_path);

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) {
    if (!trim(line).empty())
      lines.push_back(line);
  }
  return lines;
}

/// Build a label->value map from key:value lines (labels lowercased+trimmed).
/// Lines without a colon are silently skipped.
inline std::unordered_map<std::string, std::string>
build_field_map(std::span<const std::string> lines) {
  std::unordered_map<std::string, std::string> fields;
  for (const auto &line : lines) {
    auto pos = line.find(':');
    if (pos == std::string::npos)
      continue;
    auto label = to_lower(trim(line.substr(0, pos)));
    auto value = trim(line.substr(pos + 1));
    fields[label] = value;
  }
  return fields;
}

/// Get a type_real from the field map (handles Fortran notation like 1.d14).
inline type_real
get_real(const std::unordered_map<std::string, std::string> &fields,
         const std::string &key) {
  auto it = fields.find(key);
  if (it == fields.end())
    throw std::runtime_error("Missing required field: " + key);
  return static_cast<type_real>(std::stod(fortran_to_cpp_double(it->second)));
}

/// Get an int from the field map.
inline int get_int(const std::unordered_map<std::string, std::string> &fields,
                   const std::string &key) {
  auto it = fields.find(key);
  if (it == fields.end())
    throw std::runtime_error("Missing required field: " + key);
  return std::stoi(it->second);
}

} // namespace sources_impl
} // namespace io
} // namespace specfem
