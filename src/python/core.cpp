#include "specfem/simulation.hpp"
#include "specfem/simulation/context.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#include "specfem/periodic_tasks.hpp"
#include <memory>

namespace py = pybind11;

// Global ContextGuard to manage Context lifetime in Python
static std::unique_ptr<specfem::ContextGuard> global_context_guard = nullptr;

bool _initialize(py::list py_argv) {
  if (global_context_guard) {
    return false; // Already initialized
  }

  // Convert Python list to C++ string vector
  std::vector<std::string> args;
  for (const auto &item : py_argv) {
    args.push_back(item.cast<std::string>());
  }

  try {
    global_context_guard = std::make_unique<specfem::ContextGuard>(args);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool _execute(const std::string &parameter_string,
              const std::string &default_string) {
  if (global_context_guard == nullptr) {
    return false;
  }

  auto &context = global_context_guard->get_context();

  const YAML::Node parameter_dict = YAML::Load(parameter_string);
  const YAML::Node default_dict = YAML::Load(default_string);
  std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks;
  const auto signal_task =
      std::make_shared<specfem::periodic_tasks::check_signal>(10);
  tasks.push_back(signal_task);

  // Releasing the GIL in a scoped section
  // is needed for long running tasks, such as a
  // simulation.
  bool success;
  {
    py::gil_scoped_release release;
    // For now, default to 2D execution for backward compatibility
    // Later we can add a dimension parameter to the Python interface
    success =
        specfem::simulation::execute("2d", global_context_guard->get_mpi(),
                                     parameter_dict, default_dict, tasks);
  }
  return success;
}

bool _finalize() {
  if (global_context_guard == nullptr) {
    return false;
  }

  // Explicitly destroy the ContextGuard, which triggers proper cleanup
  global_context_guard.reset();
  return true;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        SPECfem++ core module
        -----------------------

        .. currentmodule:: specfempp

        .. autosummary::
           :toctree: _generate

           _run
    )pbdoc";

  m.def("_initialize", &_initialize, R"pbdoc(
        Initialize SPECFEM++.
    )pbdoc");

  m.def("_execute", &_execute, R"pbdoc(
        Execute the main SPECFEM++ workflow.
    )pbdoc");

  m.def("_finalize", &_finalize, R"pbdoc(
        Finalize SPECFEM++.
    )pbdoc");

  m.attr("_default_file_path") = __default_file__;

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
