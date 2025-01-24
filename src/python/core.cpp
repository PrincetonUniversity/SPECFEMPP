#include "execute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#include "periodic_tasks/check_signal.hpp"

namespace py = pybind11;

// global MPI variable for Python
specfem::MPI::MPI *_py_mpi = NULL;

bool _initialize(py::list py_argv) {
  if (_py_mpi != NULL) {
    return false;
  }
  // parse argc and argv from Python
  int argc = py_argv.size();
  char **argv = new char *[argc + 1];

  for (size_t i = 0; i < argc; i++) {
    std::string str =
        py_argv[i].cast<std::string>(); // Convert Python string to std::string
    argv[i] =
        new char[str.length() + 1]; // Allocate memory for each C-style string
    std::strcpy(argv[i], str.c_str()); // Copy the string content
  }

  // Null-terminate argv following the specification
  argv[argc] = nullptr;
  // Initialize MPI
  _py_mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  // free argv
  for (int i = 0; i < argc; i++) {
    delete[] argv[i]; // Free each individual string
  }

  delete[] argv;

  return true;
}

bool _execute(const std::string &parameter_string,
              const std::string &default_string) {
  if (_py_mpi == NULL) {
    return false;
  }
  const YAML::Node parameter_dict = YAML::Load(parameter_string);
  const YAML::Node default_dict = YAML::Load(default_string);
  std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks;
  const auto signal_task =
      std::make_shared<specfem::periodic_tasks::check_signal>(10);
  tasks.push_back(signal_task);
  execute(parameter_dict, default_dict, tasks, _py_mpi);
  return true;
}

bool _finalize() {
  if (_py_mpi != NULL) {
    // Finalize Kokkos
    Kokkos::finalize();
    // Finalize MPI
    delete _py_mpi;
    _py_mpi = NULL;
    return true;
  }
  return false;
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
