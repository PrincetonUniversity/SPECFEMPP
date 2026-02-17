#include "logger_options.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace specfem {
namespace logger {

LogLevel LoggerOptions::string_to_log_level(const std::string &level_str) {
  std::string upper_str = level_str;
  std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(),
                 [](unsigned char c) { return std::toupper(c); });

  if (upper_str == "TRACE")
    return LogLevel::TRACE;
  if (upper_str == "DEBUG")
    return LogLevel::DEBUG;
  if (upper_str == "INFO")
    return LogLevel::INFO;
  if (upper_str == "WARNING" || upper_str == "WARN")
    return LogLevel::WARNING;
  if (upper_str == "ERROR")
    return LogLevel::ERROR;
  if (upper_str == "CRITICAL" || upper_str == "CRIT")
    return LogLevel::CRITICAL;

  throw std::invalid_argument(
      "Invalid log level: " + level_str +
      ". Valid values: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL");
}

LoggerOptions LoggerOptions::from_values(
    std::optional<std::string> log_file, std::optional<bool> per_rank,
    std::optional<bool> auto_flush, std::optional<std::string> log_level_str) {
  LoggerOptions options;

  options.log_file = std::move(log_file);
  options.per_rank = per_rank;
  options.auto_flush = auto_flush;

  if (log_level_str.has_value()) {
    options.log_level = string_to_log_level(log_level_str.value());
  }

  return options;
}

} // namespace logger
} // namespace specfem
