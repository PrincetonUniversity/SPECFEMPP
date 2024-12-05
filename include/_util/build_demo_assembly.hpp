//=====================================NOTE=====================================
// If this code is to go into the main codebase (removed from include/_util),
// make sure to clean this up. This file is a mess that should not see the
// light of day.
//===================================END NOTE===================================

#ifndef __UTIL_DEMO_ASSEMBLY_HPP_
#define __UTIL_DEMO_ASSEMBLY_HPP_

// from specfem2d.cpp
#include "compute/interface.hpp"
// #include "coupled_interface/interface.hpp"
// #include "domain/interface.hpp"
#include "IO/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "receiver/interface.hpp"
#include "solver/solver.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "timescheme/timescheme.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
// end from specfem2d.cpp
#include "solver/time_marching.hpp"

#include "compute/fields/simulation_field.hpp"

#include "enumerations/simulation.hpp"
#include "mesh/mesh.hpp"

#include "receiver/receiver.hpp"
#include "source/source.hpp"

#include "compute/assembly/assembly.hpp"

#include "adjacency_graph/adjacency_graph.hpp"

#include "quadrature/quadrature.hpp"
#include "specfem_setup.hpp"
#include <cmath>
#include <iostream>
#include <vector>

namespace _util {
namespace demo_assembly {
constexpr specfem::dimension::type DimensionType =
    specfem::dimension::type::dim2;
void construct_demo_mesh(
    specfem::mesh::mesh<DimensionType> &mesh,
    const specfem::quadrature::quadratures &quad, const int nelemx,
    const int nelemz,
    std::vector<specfem::adjacency_graph::adjacency_pointer> &removals,
    int demo_construct_mode);

void construct_demo_mesh(
    specfem::mesh::mesh<DimensionType> &mesh,
    const specfem::quadrature::quadratures &quad,
    std::vector<specfem::adjacency_graph::adjacency_pointer> &removals,
    int demo_construct_mode);

void construct_demo_mesh(specfem::mesh::mesh<DimensionType> &mesh,
                         const specfem::quadrature::quadratures &quad,
                         const int nelemx, const int nelemz,
                         int demo_construct_mode);
void construct_demo_mesh(specfem::mesh::mesh<DimensionType> &mesh,
                         const specfem::quadrature::quadratures &quad,
                         int demo_construct_mode);

const auto _default_quadrature = []() {
  /// Gauss-Lobatto-Legendre quadrature with 5 GLL points
  const specfem::quadrature::gll::gll gll(0, 0, 5);
  return specfem::quadrature::quadratures(gll);
};

/** Builder pattern for simulation parameters.
 */
struct simulation_params {

  /** Creates a new simulation_params struct with default parameters.
   */
  simulation_params()
      : _t0(0), _dt(1), _tmax(0), _nsteps(0),
        _simulation_type(specfem::simulation::type::forward),
        _mesh(specfem::mesh::mesh<DimensionType>()),
        _quadratures(_default_quadrature()), _nseismogram_steps(0),
        _t0_adj_prio(0), _dt_adj_prio(1), _tmax_adj_prio(2), _nstep_adj_prio(3),
        needs_mesh_update(true), overwrite_nseismo_steps(true) {}

  simulation_params &t0(type_real val) {
    _t0 = val;
    _update_timevars(0);
    return *this;
  }
  simulation_params &dt(type_real val) {
    _dt = val;
    _update_timevars(1);
    return *this;
  }
  simulation_params &tmax(type_real val) {
    _tmax = val;
    _update_timevars(2);
    return *this;
  }
  simulation_params &nsteps(int val) {
    _nsteps = val;
    _update_timevars(3);
    return *this;
  }
  simulation_params &nseismogram_steps(int val) {
    overwrite_nseismo_steps = false;
    _nseismogram_steps = val;
    return *this;
  }
  simulation_params &simulation_type(specfem::simulation::type val) {
    _simulation_type = val;
    return *this;
  }
  simulation_params &mesh(specfem::mesh::mesh<DimensionType> val) {
    _mesh = val;
    needs_mesh_update = false;
    return *this;
  }
  simulation_params &quadrature(specfem::quadrature::quadratures val) {
    _quadratures = val;
    needs_mesh_update = true;
    return *this;
  }
  simulation_params &
  add_source(std::shared_ptr<specfem::sources::source> source) {
    _sources.push_back(source);
    return *this;
  }
  simulation_params &
  sources(std::vector<std::shared_ptr<specfem::sources::source> > sources) {
    _sources = sources;
    return *this;
  }
  simulation_params &
  add_receiver(std::shared_ptr<specfem::receivers::receiver> receiver) {
    _receivers.push_back(receiver);
    return *this;
  }
  simulation_params &receivers(
      std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers) {
    _receivers = receivers;
    return *this;
  }
  simulation_params &
  add_seismogram_type(specfem::enums::seismogram::type type) {
    _seismogram_types.push_back(type);
    return *this;
  }
  simulation_params &
  seismogram_types(std::vector<specfem::enums::seismogram::type> seismos) {
    _seismogram_types = seismos;
    return *this;
  }
  simulation_params &
  add_plotter(std::shared_ptr<specfem::plotter::plotter> plotter) {
    _plotters.push_back(plotter);
    return *this;
  }
  simulation_params &
  plotters(std::vector<std::shared_ptr<specfem::plotter::plotter> > plotters) {
    _plotters = plotters;
    return *this;
  }
  simulation_params &
  add_writer(std::shared_ptr<specfem::writer::writer> writer) {
    _writers.push_back(writer);
    return *this;
  }
  simulation_params &
  writers(std::vector<std::shared_ptr<specfem::writer::writer> > writers) {
    _writers = writers;
    return *this;
  }
  simulation_params &assembly(specfem::compute::assembly assembly) {
    _assembly = std::make_shared<specfem::compute::assembly>(assembly);
    return *this;
  }
  simulation_params &
  assembly(std::shared_ptr<specfem::compute::assembly> assembly) {
    _assembly = assembly;
    return *this;
  }
  simulation_params &runtime_configuration(
      std::shared_ptr<specfem::runtime_configuration::setup> runtime_config) {
    _runtime_config = runtime_config;
    return *this;
  }
  simulation_params &use_demo_mesh(int demo_construct_mode) {
    construct_demo_mesh(_mesh, _quadratures, demo_construct_mode);
    needs_mesh_update = false;
    return *this;
  }
  simulation_params &use_demo_mesh(
      int demo_construct_mode,
      std::vector<specfem::adjacency_graph::adjacency_pointer> &removals) {
    construct_demo_mesh(_mesh, _quadratures, removals, demo_construct_mode);
    needs_mesh_update = false;
    return *this;
  }
  void set_plotters_from_runtime_configuration() {
    _plotters.clear();
    if (_runtime_config) {
      _plotters.push_back(
          _runtime_config->instantiate_wavefield_plotter(*_assembly));
    }
  }
  void set_writers_from_runtime_configuration() {
    _writers.clear();
    if (_runtime_config) {
      _writers.push_back(
          _runtime_config->instantiate_seismogram_writer(*_assembly));
      _writers.push_back(
          _runtime_config->instantiate_wavefield_writer(*_assembly));
      _writers.push_back(
          _runtime_config->instantiate_kernel_writer(*_assembly));
    }
  }

  int get_numsteps() { return _nsteps; }
  int get_num_seismogram_steps() { return _nseismogram_steps; }
  type_real get_t0() { return _t0; }
  type_real get_dt() { return _dt; }
  type_real get_tmax() { return _tmax; }
  specfem::mesh::mesh<DimensionType> &get_mesh() { return _mesh; }
  std::vector<std::shared_ptr<specfem::sources::source> > &get_sources() {
    return _sources;
  }
  std::vector<std::shared_ptr<specfem::receivers::receiver> > &get_receivers() {
    return _receivers;
  }
  std::vector<std::shared_ptr<specfem::plotter::plotter> > &get_plotters() {
    return _plotters;
  }
  std::vector<std::shared_ptr<specfem::writer::writer> > &get_writers() {
    return _writers;
  }

  std::vector<specfem::enums::seismogram::type> &get_seismogram_types() {
    return _seismogram_types;
  }
  specfem::simulation::type get_simulation_type() { return _simulation_type; }

  std::shared_ptr<specfem::compute::assembly> get_assembly() {
    if (!_assembly) {
      build_assembly();
    }
    return _assembly;
  }

  void build_assembly() {
    _assembly = std::make_shared<specfem::compute::assembly>(
        _mesh, _quadratures, _sources, _receivers, _seismogram_types, _t0, _dt,
        _nsteps, _nseismogram_steps, _simulation_type);
  }

private:
  //==== for handling interdependent vars. higher #prio means it changes first.
  int8_t _t0_adj_prio;
  int8_t _dt_adj_prio;
  int8_t _tmax_adj_prio;
  int8_t _nstep_adj_prio;
  bool overwrite_nseismo_steps;
  void _update_timevars(int set_ind) {
    int8_t set_prio_prev =
        (set_ind == 0)
            ? _t0_adj_prio
            : ((set_ind == 1)
                   ? _dt_adj_prio
                   : ((set_ind == 2) ? _tmax_adj_prio : _nstep_adj_prio));
    if (set_prio_prev > _t0_adj_prio) {
      _t0_adj_prio++;
    } else if (set_prio_prev == _t0_adj_prio) {
      _t0_adj_prio = 0;
    }
    if (_t0_adj_prio == 3)
      _t0 = _tmax - _nsteps * _dt;
    if (set_prio_prev > _dt_adj_prio) {
      _dt_adj_prio++;
    } else if (set_prio_prev == _dt_adj_prio) {
      _dt_adj_prio = 0;
    }
    if (_dt_adj_prio == 3)
      _dt = (_tmax - _t0) / _nsteps;
    if (set_prio_prev > _tmax_adj_prio) {
      _tmax_adj_prio++;
    } else if (set_prio_prev == _tmax_adj_prio) {
      _tmax_adj_prio = 0;
    }
    if (_tmax_adj_prio == 3)
      _tmax = _t0 + _nsteps * _dt;
    if (set_prio_prev > _nstep_adj_prio) {
      _nstep_adj_prio++;
    } else if (set_prio_prev == _nstep_adj_prio) {
      _nstep_adj_prio = 0;
    }
    if (_nstep_adj_prio == 3)
      _nsteps = std::ceil((_tmax - _t0) / _dt);

    _nseismogram_steps = _nsteps;
  }
  //====
  type_real _t0;
  type_real _dt;
  type_real _tmax;
  int _nsteps;
  specfem::simulation::type _simulation_type;

  specfem::mesh::mesh<DimensionType> _mesh;
  specfem::quadrature::quadratures _quadratures;
  std::vector<std::shared_ptr<specfem::sources::source> > _sources;
  std::vector<std::shared_ptr<specfem::receivers::receiver> > _receivers;
  std::vector<specfem::enums::seismogram::type> _seismogram_types;
  std::vector<std::shared_ptr<specfem::plotter::plotter> > _plotters;
  std::vector<std::shared_ptr<specfem::writer::writer> > _writers;
  int _nseismogram_steps; // TODO this is max_sig_step; verify that this
                          // actually is nseismo_steps

  bool needs_mesh_update;
  std::shared_ptr<specfem::compute::assembly> _assembly;
  std::shared_ptr<specfem::runtime_configuration::setup> _runtime_config;
};

} // namespace demo_assembly
} // namespace _util
#endif
