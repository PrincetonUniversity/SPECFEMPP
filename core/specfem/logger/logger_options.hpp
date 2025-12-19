#pragma once

#include "specfem/logger.hpp"
#include <boost/program_options.hpp>
#include <optional>
#include <string>

namespace specfem {
namespace logger {

/**
 * @class LoggerOptions
 * @brief Command-line options for Logger configuration
 *
 * This class parses command-line arguments using boost::program_options
 * to allow runtime override of Logger settings.
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
   * @brief Parse command-line arguments for logger options
   *
   * @param argc Argument count from main()
   * @param argv Argument vector from main()
   * @return LoggerOptions instance with parsed values
   */
  static LoggerOptions parse(int argc, char *argv[]);

  /**
   * @brief Check if help was requested
   * @return true if --help flag was present
   */
  bool help_requested() const { return help_requested_; }

  /**
   * @brief Get the help message
   * @return Description of all available options
   */
  std::string get_help_message() const;

  // Optional values - only set if provided on command line
  std::optional<std::string> log_file; ///< Log file path
  std::optional<bool> per_rank;        ///< Per-rank file creation
  std::optional<bool> auto_flush;      ///< Auto-flush after each message
  std::optional<LogLevel> log_level;   ///< Minimum log level

private:
  LoggerOptions() = default;

  bool help_requested_ = false;
  boost::program_options::options_description desc_{ "Logger Options" };

  /**
   * @brief Convert string to LogLevel
   */
  static LogLevel string_to_log_level(const std::string &level_str);
};

} // namespace logger
} // namespace specfem
