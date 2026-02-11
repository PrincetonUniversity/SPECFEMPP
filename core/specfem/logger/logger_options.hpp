#pragma once

#include "specfem/logger/logger.hpp"
#include <optional>
#include <string>

namespace specfem {
namespace logger {

/**
 * @class LoggerOptions
 * @brief Command-line options for Logger configuration
 *
 * This class holds parsed command-line options for runtime override
 * of Logger settings.
 *
 * Supported options:
 * - --log-file=<filename>        : Set output log file
 * - --log-per-rank=<true|false>  : Enable per-rank log files and stdout
 * - --log-auto-flush=<true|false>: Enable auto-flush after each message
 * - --log-level=<level>          : Set minimum log level
 */
class LoggerOptions {
public:
  /**
   * @brief Create logger options from plain C++ values
   *
   * @param log_file Optional log file path
   * @param per_rank Optional per-rank flag
   * @param auto_flush Optional auto-flush flag
   * @param log_level_str Optional log level string
   * @return LoggerOptions instance with extracted values
   */
  static LoggerOptions from_values(std::optional<std::string> log_file,
                                   std::optional<bool> per_rank,
                                   std::optional<bool> auto_flush,
                                   std::optional<std::string> log_level_str);

  // Optional values - only set if provided on command line
  std::optional<std::string> log_file; ///< Log file path
  std::optional<bool> per_rank;        ///< Per-rank file creation
  std::optional<bool> auto_flush;      ///< Auto-flush after each message
  std::optional<LogLevel> log_level;   ///< Minimum log level

private:
  LoggerOptions() = default;

  /**
   * @brief Convert string to LogLevel
   */
  static LogLevel string_to_log_level(const std::string &level_str);
};

} // namespace logger
} // namespace specfem
