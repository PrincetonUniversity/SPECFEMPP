#ifndef PARAMETER_PARSER_H
#define PARAMETER_PARSER_H

#include "../include/config.h"
#include "../include/quadrature.h"
#include "../include/timescheme.h"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
namespace runtime_configuration {

class header {

public:
  header(std::string title, std::string description)
      : title(title), description(description){};
  header(const YAML::Node &Node);

  std::string get_title() { return this->title; }
  std::string get_description() { return this->description; }

  friend std::ostream &operator<<(std::ostream &out, header &header);

private:
  std::string title;
  std::string description;
};

class quadrature {
public:
  quadrature(type_real alpha, type_real beta, int ngllx, int ngllz)
      : alpha(alpha), beta(beta), ngllx(ngllx), ngllz(ngllz){};
  quadrature(const YAML::Node &Node);
  std::tuple<specfem::quadrature::quadrature, specfem::quadrature::quadrature>
  instantiate();

private:
  type_real alpha;
  type_real beta;
  int ngllx;
  int ngllz;
};

class solver {

public:
  virtual specfem::TimeScheme::TimeScheme *instantiate();
  virtual void update_t0(type_real t0){};
};

class time_marching : public solver {

public:
  time_marching(std::string timescheme, type_real dt, type_real nstep)
      : timescheme(timescheme), dt(dt), nstep(nstep){};
  time_marching(const YAML::Node &Node);
  void update_t0(type_real t0) override { this->t0 = t0; }
  specfem::TimeScheme::TimeScheme *instantiate() override;

private:
  int nstep;
  type_real dt;
  type_real t0;
  std::string timescheme;
};

class run_setup {

public:
  run_setup(int nproc, int nruns) : nproc(nproc), nruns(nruns){};
  run_setup(const YAML::Node &Node);

private:
  int nproc;
  int nruns;
};

class setup {

public:
  setup(std::string parameter_file);
  std::tuple<specfem::quadrature::quadrature, specfem::quadrature::quadrature>
  instantiate_quadrature() {
    return this->quadrature->instantiate();
  }
  specfem::TimeScheme::TimeScheme *instantiate_solver() {
    return this->solver->instantiate();
  }
  void update_t0(type_real t0) { this->solver->update_t0(t0); }
  std::string print_header();

private:
  specfem::runtime_configuration::header *header;
  specfem::runtime_configuration::solver *solver;
  specfem::runtime_configuration::run_setup *run_setup;
  specfem::runtime_configuration::quadrature *quadrature;
};

} // namespace runtime_configuration
} // namespace specfem

#endif
