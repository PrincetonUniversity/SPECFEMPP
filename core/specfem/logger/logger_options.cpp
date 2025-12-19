#include "logger_options.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>

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

LoggerOptions LoggerOptions::parse(int argc, char *argv[]) {
  namespace po = boost::program_options;

  LoggerOptions options;

  // Define all logger-related options
  options.desc_.add_options()("help", "Show help message")(
      "log-file", po::value<std::string>(),
      "Set output log file (base name, extensions added automatically)")(
      "log-per-rank", po::value<bool>(),
      "Enable per-rank log files and stdout for all ranks (true/false)")(
      "log-auto-flush", po::value<bool>(),
      "Enable auto-flush after each log message (true/false)")(
      "log-level", po::value<std::string>(),
      "Set minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)");

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options.desc_), vm);
    po::notify(vm);

    // Check for help request
    if (vm.count("help")) {
      options.help_requested_ = true;
      return options;
    }

    // Parse optional values
    if (vm.count("log-file")) {
      options.log_file = vm["log-file"].as<std::string>();
    }

    if (vm.count("log-per-rank")) {
      options.per_rank = vm["log-per-rank"].as<bool>();
    }

    if (vm.count("log-auto-flush")) {
      options.auto_flush = vm["log-auto-flush"].as<bool>();
    }

    if (vm.count("log-level")) {
      std::string level_str = vm["log-level"].as<std::string>();
      options.log_level = string_to_log_level(level_str);
    }

  } catch (const po::error &e) {
    std::cerr << "Error parsing command-line options: " << e.what()
              << std::endl;
    std::cerr << options.get_help_message() << std::endl;
    std::exit(1);
  } catch (const std::invalid_argument &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::cerr << options.get_help_message() << std::endl;
    std::exit(1);
  }

  return options;
}

std::string LoggerOptions::get_help_message() const {
  std::ostringstream oss;
  oss << desc_ << std::endl;
  oss << "Example usage:" << std::endl;
  oss << "  ./specfem --log-file=\"output\" --log-auto-flush=true "
         "--log-level=DEBUG"
      << std::endl;
  return oss.str();
}

} // namespace logger
} // namespace specfem
