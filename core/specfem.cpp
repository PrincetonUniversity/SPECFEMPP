#include "constants.hpp"
#include "periodic_tasks/check_signal.hpp"
#include "specfem/core.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <string>

boost::program_options::options_description define_args() {
  namespace po = boost::program_options;

  po::options_description desc{
    "======================================\n"
    "--------------- SPECFEM++ ------------\n"
    "======================================\n"
    "\n"
    "Usage: specfem <dimension> [options]\n"
    "  where <dimension> is either '2d' or '3d'\n"
  };

  desc.add_options()("help,h", "Print this help message")(
      "parameters_file,p", po::value<std::string>()->required(),
      "Location to parameters file")(
      "default_file", po::value<std::string>()->default_value(__default_file__),
      "Location of default parameters file.");

  return desc;
}

int parse_args(int argc, char **argv, boost::program_options::variables_map &vm,
               std::string &dimension) {

  const auto desc = define_args();

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

  // Get the core singleton and initialize
  auto &core = specfem::Core::instance();

  if (!core.initialize(argc, argv)) {
    std::cerr << "Failed to initialize SPECFEM++ core" << std::endl;
    return 1;
  }

  int result = 0;

  try {
    // Extract parameters (dimension is already extracted as positional
    // argument)
    const std::string parameters_file = vm["parameters_file"].as<std::string>();
    const std::string default_file = vm["default_file"].as<std::string>();

    // Load configuration files
    const YAML::Node parameter_dict = YAML::LoadFile(parameters_file);
    const YAML::Node default_dict = YAML::LoadFile(default_file);

    // Setup periodic tasks (signal checking)
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks;
    const auto signal_task =
        std::make_shared<specfem::periodic_tasks::check_signal>(10);
    tasks.push_back(signal_task);

    // Execute simulation with the specified dimension
    if (!core.execute_with_dimension(dimension, parameter_dict, default_dict,
                                     tasks)) {
      std::cerr << "Execution failed" << std::endl;
      result = 1;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error during execution: " << e.what() << std::endl;
    result = 1;
  }

  // Finalize and cleanup
  if (!core.finalize()) {
    std::cerr << "Warning: Failed to properly finalize SPECFEM++ core"
              << std::endl;
    result = 1;
  }

  return result;
}
