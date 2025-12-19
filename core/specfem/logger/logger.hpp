#pragma once

#include "specfem/mpi.hpp"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

/**
 * @file logger.hpp
 * @brief MPI-aware logging system for SPECFEM++
 *
 * This logger provides:
 * - Six log levels: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL
 * - Colored terminal output (adapts to light/dark backgrounds)
 * - File output with per-rank or single-file options
 * - Dynamic switching between file and terminal output
 * - Build-type specific defaults (Debug vs Release)
 */

// By default, only root rank prints to stdout/stderr to prevent interleaved
// output In Debug mode: all ranks can print to stdout and use per-rank files In
// Release mode: only root rank prints to stdout by default Use --log-per-rank
// runtime option to override these defaults
#ifdef NDEBUG
#define ROOT_ONLY true
#else
#define ROOT_ONLY false
#endif

namespace specfem {

// Forward declarations
namespace program {
class Context;
}

namespace logger {
class LoggerOptions;
}

/**
 * @enum LogLevel
 * @brief Enumeration of available log severity levels
 */
enum class LogLevel {
  TRACE,   ///< Detailed trace information for debugging
  DEBUG,   ///< Debug information for development
  INFO,    ///< Informational messages (default level)
  WARNING, ///< Warning messages for potential issues
  ERROR,   ///< Error messages for failures
  CRITICAL ///< Critical errors requiring immediate attention
};

/**
 * @class Logger
 * @brief Static logging class for SPECFEM++ applications
 *
 * The Logger provides a centralized, MPI-aware logging system.
 *
 * @note Must be used within a Context scope. Context manages Logger
 * initialization/finalization.
 * @note Friend class Context manages Logger lifecycle
 */
class Logger {
public:
  /**
   * @brief Set a log file for output redirection
   *
   * @param filename Base filename for the log file (extension .txt added
   * automatically)
   * @param per_rank If true, creates separate files per rank
   * @param auto_flush_override If true, forces auto-flush in Release builds
   */
  static void set_log_file(const std::string &filename, bool per_rank = false,
                           bool auto_flush_override = false);

  /**
   * @brief Set the minimum log level for filtering
   *
   * @param level Minimum log level to display
   */
  static void set_log_level(LogLevel level);

  /**
   * @brief Apply command-line options to override Logger settings
   *
   * @param options Parsed command-line options from LoggerOptions::parse()
   */
  static void apply_options(const logger::LoggerOptions &options);

  /**
   * @brief Log a trace-level message
   */
  static void trace(const std::string &message, bool root_only = ROOT_ONLY);

  /**
   * @brief Log a debug-level message
   */
  static void debug(const std::string &message, bool root_only = ROOT_ONLY);

  /**
   * @brief Log an info-level message
   */
  static void info(const std::string &message, bool root_only = ROOT_ONLY);

  /**
   * @brief Log a warning-level message
   */
  static void warning(const std::string &message, bool root_only = ROOT_ONLY);

  /**
   * @brief Log an error-level message
   */
  static void error(const std::string &message, bool root_only = ROOT_ONLY);

  /**
   * @brief Log a critical-level message
   */
  static void critical(const std::string &message, bool root_only = ROOT_ONLY);

private:
  Logger() = default;
  ~Logger() = default;
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  /**
   * @struct LogConfig
   * @brief Encapsulates all logger configuration state
   */
  struct LogConfig {
    std::ofstream log_file;        ///< Output file stream for file logging
    std::string base_filename;     ///< Base filename for log files
    bool logging_enabled = false;  ///< True when file logging is active
    bool per_rank_logging = false; ///< True when using per-rank files
    bool auto_flush = false;       ///< True to flush after each message
    LogLevel min_log_level = LogLevel::INFO; ///< Minimum level to log

    // CLI override flags
    bool cli_log_file_set = false;
    bool cli_per_rank_set = false;
    bool cli_auto_flush_set = false;
    bool cli_log_level_set = false;

    // CLI override values
    std::string cli_log_file;
    bool cli_per_rank = false;
    bool cli_auto_flush = false;
    LogLevel cli_log_level = LogLevel::INFO;
  };

  static program::Context *context_ptr_; ///< Non-owning pointer to Context
  static LogConfig config_;              ///< Logger configuration state

  /**
   * @brief Initialize Logger with Context (called by Context constructor)
   */
  static void initialize(program::Context *context);

  /**
   * @brief Finalize Logger (called by Context destructor)
   */
  static void finalize();

  /**
   * @brief Internal logging implementation
   */
  static void log_internal(LogLevel level, const std::string &message,
                           bool root_only);

  /**
   * @brief Format a log message with prefix and optional color
   */
  static std::string format_message(LogLevel level, const std::string &message,
                                    bool use_color);

  /**
   * @brief Get string representation of log level
   */
  static std::string get_level_name(LogLevel level);

  /**
   * @brief Get ANSI color code for log level
   */
  static const char *get_level_color(LogLevel level);

  /**
   * @brief Verify Logger is used within Context scope
   */
  static void check_context() {
    if (context_ptr_ == nullptr) {
      std::cerr << "ERROR: Logger used outside Context scope" << std::endl;
      std::exit(1);
    }
  }

  friend class program::Context;
};

} // namespace specfem
