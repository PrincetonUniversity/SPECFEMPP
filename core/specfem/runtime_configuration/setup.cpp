#include "setup.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <ostream>
#include <sys/stat.h>
#include <tuple>

void create_folder_if_not_exists(const std::string &folder_name) {
  struct stat info;

  if (stat(folder_name.c_str(), &info) != 0) {
    std::cout << "Creating folder: " << folder_name << std::endl;
    mkdir(folder_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  } else if (info.st_mode & S_IFDIR) {
    std::cout << "Folder already exists: " << folder_name << std::endl;
  } else {
    std::cout << "Error: " << folder_name << " is not a directory" << std::endl;
  }
}

specfem::runtime_configuration::setup::setup(const std::string &parameter_file,
                                             const std::string &default_file) {
  *this = setup(YAML::LoadFile(parameter_file), YAML::LoadFile(default_file));
}

specfem::runtime_configuration::setup::setup(const YAML::Node &parameter_dict,
                                             const YAML::Node &default_dict) {
  const YAML::Node &runtime_config = parameter_dict["parameters"];
  const YAML::Node &default_config = default_dict["default-parameters"];

  const YAML::Node &simulation_setup = runtime_config["simulation-setup"];
  const YAML::Node &n_solver = simulation_setup["solver"];

  const YAML::Node &n_databases = runtime_config["databases"];

  try {
    this->header = std::make_unique<specfem::runtime_configuration::header>(
        runtime_config["header"]);
  } catch (YAML::InvalidNode &e) {
    std::ostringstream message;
    message << "Error reading specfem parameter header. \n" << e.what();

    throw std::runtime_error(message.str());
  }

  // Get source info
  if (const YAML::Node &source_node = runtime_config["sources"]) {
    this->sources =
        std::make_unique<specfem::runtime_configuration::sources>(source_node);
  } else {
    throw std::runtime_error("Error reading specfem source configuration.");
  }

  // Get Elastic Wave type Default is P_SV
  if (const YAML::Node &n_elastic_wave = simulation_setup["elastic-wave"]) {
    this->elastic_wave =
        std::make_unique<specfem::runtime_configuration::elastic_wave>(
            n_elastic_wave);
  } else {
    this->elastic_wave =
        std::make_unique<specfem::runtime_configuration::elastic_wave>("P_SV");
  }

  // Get Electromagnetic Wave type Default is TE
  if (const YAML::Node &n_electromagnetic_wave =
          simulation_setup["electromagnetic-wave"]) {
    this->electromagnetic_wave =
        std::make_unique<specfem::runtime_configuration::electromagnetic_wave>(
            n_electromagnetic_wave);
  } else {
    this->electromagnetic_wave =
        std::make_unique<specfem::runtime_configuration::electromagnetic_wave>(
            "TE");
  }

  // Get Quadrature info
  if (const YAML::Node &n_quadrature = simulation_setup["quadrature"]) {
    this->quadrature =
        std::make_unique<specfem::runtime_configuration::quadrature>(
            n_quadrature);
  } else if (const YAML::Node &n_quadrature = default_config["quadrature"]) {
    this->quadrature =
        std::make_unique<specfem::runtime_configuration::quadrature>(
            n_quadrature);
  } else {
    throw std::runtime_error("Error reading specfem quadrature config.");
  }

  // Get Run Setup info
  if (const YAML::Node &n_run_setup = runtime_config["run-setup"]) {
    this->run_setup =
        std::make_unique<specfem::runtime_configuration::run_setup>(
            n_run_setup);
  } else if (const YAML::Node &n_run_setup = default_dict["run-setup"]) {
    this->run_setup =
        std::make_unique<specfem::runtime_configuration::run_setup>(
            n_run_setup);
  } else {
    throw std::runtime_error("Error reading specfem runtime configuration.");
  }

  // Get Database info
  try {
    this->databases = std::make_unique<
        specfem::runtime_configuration::database_configuration>(n_databases);
  } catch (YAML::InvalidNode &e) {
    std::ostringstream message;

    message << "Error reading specfem database configuration. \n" << e.what();

    throw std::runtime_error(message.str());
  }

  // Get Kernel info
  if (n_databases["writer"] && n_databases["writer"]["properties"]) {
    this->property = std::make_unique<specfem::runtime_configuration::property>(
        n_databases["writer"]["properties"], true);
  } else if (n_databases["reader"] && n_databases["reader"]["properties"]) {
    this->property = std::make_unique<specfem::runtime_configuration::property>(
        n_databases["reader"]["properties"], false);
  } else {
    this->property = nullptr;
  }

  // Get receiver info
  try {
    this->receivers =
        std::make_unique<specfem::runtime_configuration::receivers>(
            runtime_config["receivers"]);
  } catch (YAML::InvalidNode &e) {
    std::ostringstream message;
    message << "Error reading specfem receiver configuration. \n" << e.what();

    throw std::runtime_error(message.str());
  }

  // Read simulation mode node
  specfem::simulation::type simulation;
  if (const YAML::Node &n_simulation_mode =
          simulation_setup["simulation-mode"]) {
    int number_of_simulation_modes = 0;
    if (const YAML::Node &n_forward = n_simulation_mode["forward"]) {
      this->solver =
          std::make_unique<specfem::runtime_configuration::solver>("forward");
      simulation = specfem::simulation::type::forward;
      number_of_simulation_modes++;
      bool at_least_one_writer = false; // check if at least one writer is
                                        // specified
      if (const YAML::Node &n_writer = n_forward["writer"]) {
        if (const YAML::Node &n_seismogram = n_writer["seismogram"]) {
          at_least_one_writer = true;
          this->seismogram =
              std::make_unique<specfem::runtime_configuration::seismogram>(
                  n_seismogram);
        } else {
          this->seismogram = nullptr;
        }

        if (const YAML::Node &n_wavefield = n_writer["wavefield"]) {
          at_least_one_writer = true;
          this->wavefield =
              std::make_unique<specfem::runtime_configuration::wavefield>(
                  n_wavefield, specfem::simulation::type::forward);
        } else {
          this->wavefield = nullptr;
        }

        if (const YAML::Node &n_plotter = n_writer["display"]) {

#ifdef NO_VTK
          std::ostringstream message;
          message
              << "\nDisplay section is not enabled, since SPECFEM++ was built "
                 "without VTK\n"
              << "Please install VTK and rebuild SPECFEM++ with "
                 "-DVTK_DIR=/path/to/vtk\n"
              << "or\n"
              << "Disable the display section in the specfem_config.yaml "
                 "file\n";
          throw std::runtime_error(message.str());
#endif

          if ((n_plotter["simulation-field"] &&
               n_plotter["simulation-field"].as<std::string>() != "forward")) {
            std::ostringstream message;
            message << "Error: Plotting a "
                    << n_plotter["simulation-field"].as<std::string>()
                    << " wavefield in forward simulation mode. \n";
            throw std::runtime_error(message.str());
          }

          at_least_one_writer = true;
          this->plot_wavefield =
              std::make_unique<specfem::runtime_configuration::plot_wavefield>(
                  n_plotter, this->elastic_wave->get_elastic_wave_type(),
                  this->electromagnetic_wave->get_electromagnetic_wave_type());
        } else {
          this->plot_wavefield = nullptr;
        }

        this->kernel = nullptr;

        if (!at_least_one_writer) {
          throw std::runtime_error("Error in configuration file: at least one "
                                   "writer must be specified");
        }
      } else {
        throw std::runtime_error("Error in configuration file: at least one "
                                 "writer must be specified");
      }
    }

    if (const YAML::Node &n_adjoint = n_simulation_mode["combined"]) {
      this->solver =
          std::make_unique<specfem::runtime_configuration::solver>("combined");
      number_of_simulation_modes++;
      simulation = specfem::simulation::type::combined;
      if (const YAML::Node &n_reader = n_adjoint["reader"]) {
        if (const YAML::Node &n_wavefield = n_reader["wavefield"]) {
          this->wavefield =
              std::make_unique<specfem::runtime_configuration::wavefield>(
                  n_wavefield, specfem::simulation::type::combined);
        } else {
          std::ostringstream message;
          message << "Error reading adjoint reader configuration. \n"
                  << "Wavefield reader must be specified. \n";
          throw std::runtime_error(message.str());
        }
      } else {
        std::ostringstream message;
        message << "Error reading adjoint reader configuration. \n";
        throw std::runtime_error(message.str());
      }

      if (const YAML::Node &n_writer = n_adjoint["writer"]) {
        if (const YAML::Node &n_seismogram = n_writer["seismogram"]) {
          std::ostringstream message;
          message
              << "************************************************\n"
              << "Warning : Seismogram writer has been initialized for adjoint "
                 "simulation. \n"
              << "         This is generally nacessary for debugging "
                 "purposes. \n"
              << "         If this is a production run then reconsider if "
                 "seismogram computation is needed. \n"
              << "************************************************\n";
          std::cout << message.str();
          this->seismogram =
              std::make_unique<specfem::runtime_configuration::seismogram>(
                  n_seismogram);
        } else {
          this->seismogram = nullptr;
        }

        if (const YAML::Node &n_kernel = n_writer["kernels"]) {
          this->kernel =
              std::make_unique<specfem::runtime_configuration::kernel>(
                  n_kernel, specfem::simulation::type::combined);
        } else {
          std::ostringstream message;
          message << "Error reading adjoint writer configuration. \n"
                  << "Kernel writer must be specified. \n";

          throw std::runtime_error(message.str());
        }

        if (const YAML::Node &n_plotter = n_writer["display"]) {
          if (n_plotter["simulation-field"] &&
              specfem::utilities::is_forward_string(
                  n_plotter["simulation-field"].as<std::string>())) {
            std::ostringstream message;
            message << "Error: Plotting a forward wavefield in combined "
                    << "simulation mode. \n";
            throw std::runtime_error(message.str());
          }
          this->plot_wavefield =
              std::make_unique<specfem::runtime_configuration::plot_wavefield>(
                  n_plotter, this->elastic_wave->get_elastic_wave_type(),
                  this->electromagnetic_wave->get_electromagnetic_wave_type());
        } else {
          this->plot_wavefield = nullptr;
        }
      }
    }

    if (number_of_simulation_modes != 1) {
      throw std::runtime_error("Error in configuration file: exactly one "
                               "simulation mode must be specified");
    }
  } else {
    throw std::runtime_error("Error reading specfem simulation mode.");
  }

  try {
    const YAML::Node &n_time_marching = n_solver["time-marching"];
    const YAML::Node &n_timescheme = n_time_marching["time-scheme"];

    this->time_scheme =
        std::make_unique<specfem::runtime_configuration::time_scheme>(
            n_timescheme, simulation);
  } catch (YAML::InvalidNode &e) {
    std::ostringstream message;
    message << "Error reading specfem solver configuration. \n" << e.what();
    throw std::runtime_error(message.str());
  }
}

// Explicit template instantiations for instantiate_timescheme
template std::shared_ptr<specfem::time_scheme::time_scheme>
specfem::runtime_configuration::setup::instantiate_timescheme<
    specfem::assembly::fields<specfem::dimension::type::dim2> >(
    specfem::assembly::fields<specfem::dimension::type::dim2> &fields) const;

template std::shared_ptr<specfem::time_scheme::time_scheme>
specfem::runtime_configuration::setup::instantiate_timescheme<
    specfem::assembly::fields<specfem::dimension::type::dim3> >(
    specfem::assembly::fields<specfem::dimension::type::dim3> &fields) const;
