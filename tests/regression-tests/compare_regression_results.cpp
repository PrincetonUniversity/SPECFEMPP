#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <boost/program_options.hpp>

boost::program_options::options_description define_args() {
  namespace po = boost::program_options;

  po::options_description desc{
    "===================================================\n"
    "------------Checking regression results------------\n"
    "==================================================="
  };

  desc.add_options()("help,h", "Print this help message")(
      "PR", po::value<std::string>(),
      "YAML file storing regression test result from PR branch")(
      "main", po::value<std::string>(),
      "YAML file storing regression test result from main branch")(
      "threshold", po::value<type_real>()->default_value(0.95),
      "Threshold value to pass the test. If the performance falls below this "
      "value then the test fails.");

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

  if (!vm.count("PR")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (!vm.count("main")) {
    std::cout << desc << std::endl;
    return 0;
  }

  return 1;
}

int run_test(const std::string &PR_regression_results_file,
             const std::string &main_regression_results_file,
             const type_real threshhold) {
  const YAML::Node PR_regression_results =
      YAML::LoadFile(PR_regression_results_file);
  const YAML::Node main_regression_results =
      YAML::LoadFile(main_regression_results_file);

  const YAML::Node PR_results = PR_regression_results["results"];
  const YAML::Node main_results = main_regression_results["results"];

  assert(PR_results.IsSequence());
  assert(main_results.IsSequence());

  std::ostringstream message;

  message << "===================================================\n"
          << "------------Checking regression results------------\n"
          << "===================================================";

  std::cout << message.str() << std::endl;

  for (YAML::const_iterator it = main_results.begin(); it != main_results.end();
       ++it) {
    std::string test_name = it->first.as<std::string>();
    type_real value = it->second.as<type_real>();

    if (PR_results[test_name]) {
      if (value / PR_results[test_name].as<type_real>() < threshhold) {
        message.clear();
        message << "Performance for test : " << test_name
                << " not within limits.\n"
                << "    Test performance on main branch : " << value << "\n"
                << "    Test performance on PR branch : "
                << PR_results[test_name].as<type_real>();
        throw std::runtime_error(message.str());
      } else {
        message.clear();
        message << test_name << " ........... "
                << "PASSED";
        std::cout << message.str() << std::endl;
      }
    }
  }

  message.clear();

  message << "===================================================\n"
          << "-----------------------Done------------------------\n"
          << "===================================================";

  std::cout << message.str() << std::endl;

  return 0;
}

int main(int argc, char **argv) {
  boost::program_options::variables_map vm;
  if (parse_args(argc, argv, vm)) {
    const std::string PR_regression_results_file = vm["PR"].as<std::string>();
    const std::string main_regression_results_file =
        vm["main"].as<std::string>();
    const type_real threshold = vm["threshold"].as<type_real>();
    return run_test(PR_regression_results_file, main_regression_results_file,
                    threshold);
  }

  return 0;
}
