#include "specfem/program.hpp"
#include "dim2/program.hpp"
#include "dim3/program.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh.hpp"
#include "specfem/program.tpp"
#include "specfem/receivers.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/setup.hpp"
#include "specfem/solver.hpp"
#include "specfem/source.hpp"
#include "specfem/timescheme.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"

#include <Kokkos_Core.hpp>
#include <sstream>

namespace specfem::program {

std::string
print_end_message(std::chrono::time_point<std::chrono::system_clock> start_time,
                  std::chrono::duration<double> solver_time) {
  std::ostringstream message;
  // current date/time based on current system
  const auto now = std::chrono::system_clock::now();

  std::time_t c_now = std::chrono::system_clock::to_time_t(now);

  std::chrono::duration<double> diff = now - start_time;

  message << "\n================================================\n"
          << "             Finished simulation\n"
          << "================================================\n\n"
          << "Total simulation time : " << diff.count() << " secs\n"
          << "Total solver time (time loop) : " << solver_time.count()
          << " secs\n"
          << "Simulation end time : " << ctime(&c_now)
          << "------------------------------------------------\n";

  return message.str();
}

bool execute(const std::string &dimension, const YAML::Node &parameter_dict) {
  try {
    // Use simulation model enumeration for validation
    specfem::simulation::model simulation_model =
        specfem::simulation::from_string(dimension);

    switch (simulation_model) {
    case specfem::simulation::model::Cartesian2D: {
      // Setup periodic tasks (signal checking)
      const auto dimension_tag = specfem::element::dimension_tag::dim2;
      std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<dimension_tag>>>
          tasks;
      const auto signal_task = std::make_shared<
          specfem::periodic_tasks::check_signal<dimension_tag>>(10);
      tasks.push_back(signal_task);

      // Run 2D Cartesian program
      program_2d(parameter_dict, tasks);

      return true;
    }
    case specfem::simulation::model::Cartesian3D: {
      // Setup periodic tasks (signal checking)
      const auto dimension_tag = specfem::element::dimension_tag::dim3;
      std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<dimension_tag>>>
          tasks;
      const auto signal_task = std::make_shared<
          specfem::periodic_tasks::check_signal<dimension_tag>>(10);
      tasks.push_back(signal_task);

      // Run 3D Cartesian program
      program_3d(parameter_dict, tasks);

      return true;
    }
    case specfem::simulation::model::Globe3D: {
      const auto dimension_tag = specfem::element::dimension_tag::dim3;
      std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<dimension_tag>>>
          tasks;
      tasks.push_back(
          std::make_shared<
              specfem::periodic_tasks::check_signal<dimension_tag>>(10));
      program_3d(parameter_dict, tasks, true);
      return true;
    }
    default: {
      specfem::Logger::error("Unsupported simulation model");
      return false;
    }
    }
  } catch (const std::exception &e) {
    specfem::Logger::error(std::string("Error during execution: ") + e.what());
    return false;
  }
}

} // namespace specfem::program

template std::string
specfem::program::print_header<specfem::element::dimension_tag::dim2>(
    const specfem::runtime_configuration::setup &,
    const std::chrono::time_point<std::chrono::system_clock>);
template std::string
specfem::program::print_header<specfem::element::dimension_tag::dim3>(
    const specfem::runtime_configuration::setup &,
    const std::chrono::time_point<std::chrono::system_clock>);
