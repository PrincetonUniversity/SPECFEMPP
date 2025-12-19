#include "constants.hpp"
#include "specfem/periodic_tasks.hpp"
#include "specfem/program.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <yaml-cpp/yaml.h>

boost::program_options::options_description define_args() {
  namespace po = boost::program_options;

  po::options_description desc{ "======================================\n"
                                "------------SPECFEM Kokkos------------\n"
                                "======================================" };

  desc.add_options()("help,h", "Print this help message")(
      "parameters_file,p", po::value<std::string>(),
      "Location to parameters file")(
      "default_file,d",
      po::value<std::string>()->default_value(__default_file__),
      "Location of default parameters file.");

  return desc;
}

int parse_args(int argc, char **argv,
               boost::program_options::variables_map &vm) {

  const auto desc = define_args();
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (!vm.count("parameters_file")) {
    std::cout << desc << std::endl;
    return 0;
  }

  return 1;
}

int main(int argc, char **argv) {
  // Parse command line arguments
  boost::program_options::variables_map vm;
  int parse_result = parse_args(argc, argv, vm);

  if (parse_result <= 0) {
    return (parse_result == 0) ? 0 : 1; // 0 for help, 1 for error
  }

  // Use Context for automatic RAII-based initialization and cleanup
  int result = 0;

  try {
    // Initialize context with RAII
    specfem::program::Context context(argc, argv);

    // Extract parameters
    const std::string parameters_file = vm["parameters_file"].as<std::string>();
    const std::string default_file = vm["default_file"].as<std::string>();

    // Load configuration files
    const YAML::Node parameter_dict = YAML::LoadFile(parameters_file);
    const YAML::Node default_dict = YAML::LoadFile(default_file);

    // Execute program for 3D
    const auto success = specfem::program::execute(
        "3d", context.get_mpi(), parameter_dict, default_dict);

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
