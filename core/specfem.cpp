#include "specfem/constants.hpp"
#include "specfem/logger.hpp"
#include "specfem/program.hpp"
#include "specfem/program/context.hpp"
#include <CLI/CLI.hpp>
#include <iostream>
#include <optional>
#include <string>

// Options shared by simulation subcommands (2d, 3d)
struct SimulationOptions {
  std::string parameters_file;
  std::string log_file;
  bool log_per_rank = false;
  bool log_auto_flush = false;
  std::string log_level;
};

// Flags tracking which logger options were explicitly set on the CLI
struct LoggerFlags {
  bool log_file_set = false;
  bool per_rank_set = false;
  bool auto_flush_set = false;
  bool log_level_set = false;
};

void add_logger_options(CLI::App *cmd, SimulationOptions &opts,
                        LoggerFlags &flags) {
  cmd->add_option("--log-file", opts.log_file,
                  "Set output log file (base name, '.log' extension added "
                  "automatically)")
      ->each([&](const std::string &) { flags.log_file_set = true; });

  cmd->add_option("--log-per-rank", opts.log_per_rank,
                  "Enable per-rank log files and stdout for all ranks "
                  "(true/false)")
      ->each([&](const std::string &) { flags.per_rank_set = true; });

  cmd->add_option("--log-auto-flush", opts.log_auto_flush,
                  "Enable auto-flush after each log message (true/false)")
      ->each([&](const std::string &) { flags.auto_flush_set = true; });

  cmd->add_option("--log-level", opts.log_level,
                  "Set minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR, "
                  "CRITICAL)")
      ->each([&](const std::string &) { flags.log_level_set = true; });
}

void add_simulation_options(CLI::App *cmd, SimulationOptions &opts,
                            LoggerFlags &flags) {
  cmd->add_option("-p,--parameters-file", opts.parameters_file,
                  "Location to parameters file")
      ->required();
  add_logger_options(cmd, opts, flags);
}

int run_simulation(const std::string &dimension, int argc, char **argv,
                   const SimulationOptions &opts, const LoggerFlags &flags) {
  int result = 0;

  try {
    specfem::program::Context context(argc, argv);

    // Only run simulation if current rank is part of the subset communicator
    // (i.e. not excluded by user-defined nprocs)
    // For simulations that use the entire world communicator, this check will
    // pass for all ranks.
    if (specfem::MPI::communicator() != MPI_COMM_NULL) {

      const YAML::Node parameter_dict = YAML::LoadFile(opts.parameters_file);

      // Build LoggerOptions from CLI values
      std::optional<std::string> log_file_opt;
      std::optional<bool> per_rank_opt;
      std::optional<bool> auto_flush_opt;
      std::optional<std::string> log_level_opt;

      if (flags.log_file_set)
        log_file_opt = opts.log_file;
      if (flags.per_rank_set)
        per_rank_opt = opts.log_per_rank;
      if (flags.auto_flush_set)
        auto_flush_opt = opts.log_auto_flush;
      if (flags.log_level_set)
        log_level_opt = opts.log_level;

      auto logger_options = specfem::logger::LoggerOptions::from_values(
          std::move(log_file_opt), per_rank_opt, auto_flush_opt,
          std::move(log_level_opt));
      specfem::Logger::apply_options(logger_options);

      // Set log file if specified in parameters and not already set by CLI
      if (parameter_dict["parameters"]["log-file"]) {
        const std::string log_file =
            parameter_dict["parameters"]["log-file"].as<std::string>();
        specfem::Logger::set_log_file(log_file);
      }

      const auto success = specfem::program::execute(dimension, parameter_dict);

      if (!success) {
        std::cerr << "Execution failed. See logs for details." << std::endl;
        result = 1;
      }
    } else {
      result = 0;
    }

    SPECFEM_MPI_SAFECALL(MPI_Barrier(MPI_COMM_WORLD));

    // Ensure all ranks have reached this point, reduce the result across ranks
    int result_reduced = result;
    SPECFEM_MPI_SAFECALL(MPI_Allreduce(&result, &result_reduced, 1, MPI_INT,
                                       MPI_MAX, MPI_COMM_WORLD));
    result = result_reduced;

  } catch (const std::exception &e) {
    std::cerr << "Error during execution: " << e.what() << std::endl;
    result = 1;
  }

  return result;
}

struct Qoptions {
  type_real Q;
  type_real minfreq;
  type_real maxfreq;
  type_real min_plot_freq;
  type_real max_plot_freq;
  std::string output_dir;
};

void add_qplot_options(CLI::App *cmd, Qoptions &opts) {
  cmd->add_option("Q", opts.Q, "Quality factor Q")->required();
  cmd->add_option("minfreq", opts.minfreq, "Minimum frequency")->required();
  cmd->add_option("maxfreq", opts.maxfreq, "Maximum frequency")->required();
  cmd->add_option("min_plot_freq", opts.min_plot_freq, "Minimum plot frequency")
      ->capture_default_str()
      ->default_val(1e-5);
  cmd->add_option("max_plot_freq", opts.max_plot_freq, "Maximum plot frequency")
      ->capture_default_str()
      ->default_val(1e2);
  cmd->add_option("--output-dir", opts.output_dir,
                  "Directory to save Q plot data")
      ->capture_default_str()
      ->default_val("qplots");
}

int run_qplots(int argc, char **argv, const Qoptions &opts) {

  int result = 0;

  try {
    specfem::program::Context context(argc, argv);

    // Only run Q plot generation if current rank is part of the subset
    // communicator (i.e. not excluded by user-defined nprocs). For simulations
    // that use the entire world communicator, this check will pass for all
    // ranks.
    if (specfem::MPI::communicator() != MPI_COMM_NULL) {

      const auto success = specfem::program::qplots(
          opts.Q, opts.minfreq, opts.maxfreq, opts.min_plot_freq,
          opts.max_plot_freq, opts.output_dir);

      if (!success) {
        std::cerr << "Q plot generation failed. See logs for details."
                  << std::endl;
        result = 1;
      }
    } else {
      result = 0;
    }

    SPECFEM_MPI_SAFECALL(MPI_Barrier(MPI_COMM_WORLD));

    // Ensure all ranks have reached this point, reduce the result across ranks
    int result_reduced = result;
    SPECFEM_MPI_SAFECALL(MPI_Allreduce(&result, &result_reduced, 1, MPI_INT,
                                       MPI_MAX, MPI_COMM_WORLD));
    result = result_reduced;

  } catch (const std::exception &e) {
    std::cerr << "Error during Q plot generation: " << e.what() << std::endl;
    result = 1;
  }

  return result;
}

int main(int argc, char **argv) {

  CLI::App app{ "======================================\n"
                "--------------- SPECFEM++ ------------\n"
                "======================================" };
  app.require_subcommand(1);

  // -- 2d subcommand --
  SimulationOptions opts_2d;
  LoggerFlags flags_2d;
  auto *cmd_2d = app.add_subcommand("2d", "Run 2D simulation");
  add_simulation_options(cmd_2d, opts_2d, flags_2d);

  // -- 3d subcommand --
  SimulationOptions opts_3d;
  LoggerFlags flags_3d;
  auto *cmd_3d = app.add_subcommand("3d", "Run 3D simulation");
  add_simulation_options(cmd_3d, opts_3d, flags_3d);

  // -- Qplots subcommand (placeholder) --
  Qoptions qplots_opts;
  auto *cmd_qplots =
      app.add_subcommand("qplots", "Generate Q attenuation plots");
  add_qplot_options(cmd_qplots, qplots_opts);

  CLI11_PARSE(app, argc, argv);

  // Dispatch
  if (cmd_2d->parsed()) {
    return run_simulation("2d", argc, argv, opts_2d, flags_2d);
  }

  if (cmd_3d->parsed()) {
    return run_simulation("3d", argc, argv, opts_3d, flags_3d);
  }

  if (cmd_qplots->parsed()) {
    return run_qplots(argc, argv, qplots_opts);
  }

  return 1;
}
