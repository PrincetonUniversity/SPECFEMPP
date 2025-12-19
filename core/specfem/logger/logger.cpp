#include "logger.hpp"
#include "logger_options.hpp"
#include <algorithm>
#include <sstream>

// ANSI color codes - bright versions for visibility
#define COLOR_RESET "\033[0m"
#define COLOR_TRACE "\033[90m"    // Bright Black (Gray)
#define COLOR_DEBUG "\033[96m"    // Bright Cyan
#define COLOR_INFO "\033[0m"      // Default terminal color
#define COLOR_WARNING "\033[93m"  // Bright Yellow
#define COLOR_ERROR "\033[91m"    // Bright Red
#define COLOR_CRITICAL "\033[95m" // Bright Magenta

namespace specfem {

program::Context *Logger::context_ptr_ = nullptr;
Logger::LogConfig Logger::config_;

void Logger::initialize(program::Context *context) {
  if (context_ptr_) {
    std::cerr << "ERROR: Logger already initialized" << std::endl;
    std::exit(1);
  }
  context_ptr_ = context;
}

void Logger::finalize() {
  if (config_.log_file.is_open()) {
    config_.log_file.close();
  }
  config_.logging_enabled = false;
  config_.per_rank_logging = false;
  context_ptr_ = nullptr;
}

void Logger::set_log_file(const std::string &filename, bool per_rank,
                          bool auto_flush_override) {
  check_context();

  if (filename.empty()) {
    std::cerr << "ERROR: Cannot set log file with empty filename" << std::endl;
    std::exit(1);
  }

  if (config_.log_file.is_open()) {
    std::string msg = "Log file already set";
    if (config_.cli_log_file_set) {
      msg += " via command line (--log-file)";
    } else {
      msg += " programmatically";
    }
    msg += ". Ignoring this call to set_log_file().";
    warning(msg);
    return;
  }

  // Apply CLI overrides if they were set
  std::string actual_filename =
      config_.cli_log_file_set ? config_.cli_log_file : filename;
  bool actual_per_rank =
      config_.cli_per_rank_set ? config_.cli_per_rank : per_rank;
  bool actual_auto_flush =
      config_.cli_auto_flush_set ? config_.cli_auto_flush : auto_flush_override;

  config_.base_filename = actual_filename;

#ifndef NDEBUG
  // Debug mode: auto-flush always on, per-rank files by default
  config_.auto_flush = true;
  if (!actual_per_rank) {
    actual_per_rank = true;
  }
#else
  // Release mode: auto-flush off by default, single file by default
  config_.auto_flush = actual_auto_flush;
#endif

  config_.per_rank_logging = actual_per_rank;

  std::string output_filename = actual_filename;

  if (actual_per_rank) {
    std::ostringstream rank_str;
    rank_str << std::setfill('0') << std::setw(5) << MPI_new::rank;
    output_filename += ".rank" + rank_str.str();
  } else {
    if (MPI_new::rank != 0) {
      config_.logging_enabled = false;
      return;
    }
  }

  output_filename += ".log";

  config_.log_file.open(output_filename, std::ios::out | std::ios::trunc);

  if (!config_.log_file.is_open()) {
    throw std::runtime_error("Failed to open log file: " + output_filename);
  }

  config_.logging_enabled = true;
}

void Logger::set_log_level(LogLevel level) {
  // CLI options take precedence
  if (config_.cli_log_level_set) {
    return;
  }
  config_.min_log_level = level;
}

void Logger::apply_options(const logger::LoggerOptions &options) {
  check_context();

  // Store CLI-provided log level
  if (options.log_level.has_value()) {
    config_.cli_log_level = options.log_level.value();
    config_.cli_log_level_set = true;
    config_.min_log_level = config_.cli_log_level;
  }

  // Store CLI-provided log file settings
  if (options.log_file.has_value()) {
    config_.cli_log_file = options.log_file.value();
    config_.cli_log_file_set = true;
  }

  if (options.per_rank.has_value()) {
    config_.cli_per_rank = options.per_rank.value();
    config_.cli_per_rank_set = true;
  }

  if (options.auto_flush.has_value()) {
    config_.cli_auto_flush = options.auto_flush.value();
    config_.cli_auto_flush_set = true;
  }

  // If log file was specified via CLI, apply it immediately
  if (config_.cli_log_file_set) {
    bool per_rank = config_.cli_per_rank_set ? config_.cli_per_rank : false;
    bool auto_flush =
        config_.cli_auto_flush_set ? config_.cli_auto_flush : false;
    set_log_file(config_.cli_log_file, per_rank, auto_flush);
  }
}

std::string Logger::get_level_name(LogLevel level) {
  switch (level) {
  case LogLevel::TRACE:
    return "TRACE";
  case LogLevel::DEBUG:
    return "DEBUG";
  case LogLevel::INFO:
    return "INFO";
  case LogLevel::WARNING:
    return "WARNING";
  case LogLevel::ERROR:
    return "ERROR";
  case LogLevel::CRITICAL:
    return "CRITICAL";
  default:
    return "UNKNOWN";
  }
}

const char *Logger::get_level_color(LogLevel level) {
  switch (level) {
  case LogLevel::TRACE:
    return COLOR_TRACE;
  case LogLevel::DEBUG:
    return COLOR_DEBUG;
  case LogLevel::INFO:
    return COLOR_INFO;
  case LogLevel::WARNING:
    return COLOR_WARNING;
  case LogLevel::ERROR:
    return COLOR_ERROR;
  case LogLevel::CRITICAL:
    return COLOR_CRITICAL;
  default:
    return COLOR_RESET;
  }
}

std::string Logger::format_message(LogLevel level, const std::string &message,
                                   bool use_color) {
  std::ostringstream result;

  std::string prefix;
  std::ostringstream prefix_oss;

  std::string level_name = get_level_name(level);
  int padding = 8 - level_name.length();
  int left_pad = padding / 2;
  int right_pad = padding - left_pad;

  prefix_oss << "[SF++][" << std::string(left_pad, ' ') << level_name
             << std::string(right_pad, ' ') << "]";

  // Determine if we should show rank numbers
  bool show_rank = false;
#ifndef NDEBUG
  show_rank = true;
#endif
  // CLI option overrides build-type default
  if (config_.cli_per_rank_set) {
    show_rank = config_.cli_per_rank;
  }

  if (show_rank) {
    prefix_oss << "[>" << std::setfill('0') << std::setw(5) << MPI_new::rank
               << "<]: ";
  } else {
    prefix_oss << ": ";
  }

  prefix = prefix_oss.str();

  std::istringstream msg_stream(message);
  std::string line;
  bool first_line = true;

  while (std::getline(msg_stream, line)) {
    if (!first_line) {
      result << "\n";
    }

    if (use_color) {
      result << get_level_color(level);
    }

    result << prefix << line;

    if (use_color) {
      result << COLOR_RESET;
    }

    first_line = false;
  }

  return result.str();
}

void Logger::log_internal(LogLevel level, const std::string &message,
                          bool root_only) {
  check_context();

  if (level < config_.min_log_level) {
    return;
  }

  std::string file_output = format_message(level, message, false);
  std::string stdout_output = format_message(level, message, true);

  if (config_.logging_enabled) {
    if (config_.log_file.is_open()) {
      config_.log_file << file_output << std::endl;
      if (config_.auto_flush) {
        config_.log_file.flush();
      }
    }
  } else {
    // Check if CLI per_rank option should override build-type behavior
    bool allow_all_ranks_stdout = false;
#ifndef NDEBUG
    allow_all_ranks_stdout = true;
#endif
    // CLI option overrides build-type default
    if (config_.cli_per_rank_set) {
      allow_all_ranks_stdout = config_.cli_per_rank;
    }

    if (allow_all_ranks_stdout) {
      if (root_only && MPI_new::rank != 0) {
        return;
      }
    } else {
      if (MPI_new::rank != 0) {
        return;
      }
    }

    if (level >= LogLevel::ERROR) {
      std::cerr << stdout_output << std::endl;
    } else {
      std::cout << stdout_output << std::endl;
    }
  }
}

void Logger::trace(const std::string &message, bool root_only) {
  log_internal(LogLevel::TRACE, message, root_only);
}

void Logger::debug(const std::string &message, bool root_only) {
  log_internal(LogLevel::DEBUG, message, root_only);
}

void Logger::info(const std::string &message, bool root_only) {
  log_internal(LogLevel::INFO, message, root_only);
}

void Logger::warning(const std::string &message, bool root_only) {
  log_internal(LogLevel::WARNING, message, root_only);
}

void Logger::error(const std::string &message, bool root_only) {
  log_internal(LogLevel::ERROR, message, root_only);
}

void Logger::critical(const std::string &message, bool root_only) {
  log_internal(LogLevel::CRITICAL, message, root_only);
}

} // namespace specfem
