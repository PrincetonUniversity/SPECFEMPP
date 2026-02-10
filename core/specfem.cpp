#include "specfem/constants.hpp"
#include "specfem/logger.hpp"
#include "specfem/program.hpp"
#include "specfem/program/context.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <string>

/**
 * @brief Define command line options for SPECFEM++ executable
 * @return Boost program options description
 *
 * @code{.sh}
 * specfem --help
 * @endcode
 */
boost::program_options::options_description define_specfem_args() {
  namespace po = boost::program_options;

  po::options_description desc{
    "======================================\n"
    "--------------- SPECFEM++ ------------\n"
    "======================================\n"
    "\n"
    "Usage: specfem <dimension> [options]\n"
    "  where <dimension> is either '2d' or '3d'\n"
  };

  // Add basic options
  desc.add_options()("help,h", "Print this help message")(
      "parameters_file,p", po::value<std::string>()->required(),
      "Location to parameters file");

  // Add logger options
  desc.add_options()(
      "log-file", po::value<std::string>(),
      "Set output log file (base name, '.log' extension added automatically)")(
      "log-per-rank", po::value<bool>(),
      "Enable per-rank log files and stdout for all ranks (true/false)")(
      "log-auto-flush", po::value<bool>(),
      "Enable auto-flush after each log message (true/false)")(
      "log-level", po::value<std::string>(),
      "Set minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)");

  return desc;
}

/**
 * @brief Parse and validate command line arguments
 * @param argc Argument count
 * @param argv Argument vector
 * @param vm Variables map to store parsed arguments
 * @param dimension Output parameter for dimension (2d/3d)
 * @return 1 if successful, 0 if help requested, -1 if error
 */
int parse_args(int argc, char **argv, boost::program_options::variables_map &vm,
               std::string &dimension) {

  const auto desc = define_specfem_args();

  try {
    // Check for minimum arguments (program name + dimension)
    if (argc < 2) {
      std::cout << desc << std::endl;
      return 0;
    }

    // Check for help first
    if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
      std::cout << desc << std::endl;
      return 0;
    }

    // Extract dimension as positional argument
    dimension = std::string(argv[1]);

    // Validate dimension argument
    if (dimension != "2d" && dimension != "3d" && dimension != "dim2" &&
        dimension != "dim3") {
      std::cerr << "Error: Invalid dimension '" << dimension
                << "'. Use '2d' or '3d'." << std::endl;
      std::cout << desc << std::endl;
      return -1;
    }

    // Parse remaining arguments (skip program name and dimension)
    boost::program_options::store(
        boost::program_options::parse_command_line(argc - 1, argv + 1, desc),
        vm);

    boost::program_options::notify(vm);

    return 1;
  } catch (const boost::program_options::error &e) {
    std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    std::cout << desc << std::endl;
    return -1;
  }
}

int main(int argc, char **argv) {
  // Parse command line arguments
  boost::program_options::variables_map vm;
  std::string dimension;
  int parse_result = parse_args(argc, argv, vm, dimension);

  if (parse_result <= 0) {
    return (parse_result == 0) ? 0 : 1; // 0 for help, 1 for error
  }

  // Use Context for automatic RAII-based initialization and cleanup
  int result = 0;

  try {
    // Initialize context with RAII
    specfem::program::Context context(argc, argv);

    // Extract parameters (dimension is already extracted as positional
    // argument)
    const std::string parameters_file = vm["parameters_file"].as<std::string>();

    // Load configuration file
    const YAML::Node parameter_dict = YAML::LoadFile(parameters_file);

    // Extract and apply Logger options from parsed arguments
    auto logger_options =
        specfem::logger::LoggerOptions::from_variables_map(vm);
    specfem::Logger::apply_options(logger_options);

    // Set log file if specified in parameters and not already set by CLI
    if (parameter_dict["parameters"]["log-file"]) {
      const std::string log_file =
          parameter_dict["parameters"]["log-file"].as<std::string>();
      specfem::Logger::set_log_file(log_file);
    }

    // Execute program with the specified dimension
    const auto success =
        specfem::program::execute(dimension, parameter_dict);

    // Check execution result
    if (!success) {
      std::cerr << "Execution failed" << std::endl;
      result = 1;
    }

    // Context automatically finalized when guard goes out of scope

  } catch (const std::exception &e) {
    std::cerr << "Error during execution: " << e.what() << std::endl;
    result = 1;
  }

  return result;
}
