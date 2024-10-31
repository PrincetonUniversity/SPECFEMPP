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
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
// #include "parameter_parser/interface.hpp"
#include "receiver/interface.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
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

void construct_demo_mesh(
    specfem::mesh::mesh &mesh, const specfem::quadrature::quadratures &quad,
    const int nelemx, const int nelemz,
    std::vector<specfem::adjacency_graph::adjacency_pointer> &removals);

void construct_demo_mesh(
    specfem::mesh::mesh &mesh, const specfem::quadrature::quadratures &quad,
    std::vector<specfem::adjacency_graph::adjacency_pointer> &removals);

void construct_demo_mesh(specfem::mesh::mesh &mesh,
                         const specfem::quadrature::quadratures &quad,
                         const int nelemx, const int nelemz);
void construct_demo_mesh(specfem::mesh::mesh &mesh,
                         const specfem::quadrature::quadratures &quad);

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
        _mesh(specfem::mesh::mesh()), _quadratures(_default_quadrature()),
        _sources(std::vector<std::shared_ptr<specfem::sources::source> >()),
        _receivers(
            std::vector<std::shared_ptr<specfem::receivers::receiver> >()),
        _seismogram_types(std::vector<specfem::enums::seismogram::type>()),
        _nseismogram_steps(0), _t0_adj_prio(0), _dt_adj_prio(1),
        _tmax_adj_prio(2), _nstep_adj_prio(3), needs_mesh_update(true) {}

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
  simulation_params &simulation_type(specfem::simulation::type val) {
    _simulation_type = val;
    return *this;
  }
  simulation_params &mesh(specfem::mesh::mesh val) {
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
  add_receiver(std::shared_ptr<specfem::receivers::receiver> receiver) {
    _receivers.push_back(receiver);
    return *this;
  }
  simulation_params &
  add_seismogram_type(specfem::enums::seismogram::type type) {
    _seismogram_types.push_back(type);
    return *this;
  }
  simulation_params &use_demo_mesh() {
    construct_demo_mesh(_mesh, _quadratures);
    needs_mesh_update = false;
    return *this;
  }
  simulation_params &use_demo_mesh(
      std::vector<specfem::adjacency_graph::adjacency_pointer> &removals) {
    construct_demo_mesh(_mesh, _quadratures, removals);
    needs_mesh_update = false;
    return *this;
  }
  int get_numsteps() { return _nsteps; }
  int get_num_seismogram_steps() { return _nseismogram_steps; }
  type_real get_t0() { return _t0; }
  type_real get_dt() { return _dt; }
  type_real get_tmax() { return _tmax; }
  specfem::mesh::mesh &get_mesh() { return _mesh; }
  std::vector<std::shared_ptr<specfem::sources::source> > &get_sources() {
    return _sources;
  }
  std::vector<std::shared_ptr<specfem::receivers::receiver> > &get_receivers() {
    return _receivers;
  }

  std::vector<specfem::enums::seismogram::type> &get_seismogram_types() {
    return _seismogram_types;
  }
  specfem::simulation::type get_simulation_type() { return _simulation_type; }

  specfem::compute::assembly build_assembly() {
    specfem::compute::assembly assembly(
        _mesh, _quadratures, _sources, _receivers, _seismogram_types, _t0, _dt,
        _nsteps, _nseismogram_steps, _simulation_type);
    return assembly;
  }

private:
  //==== for handling interdependent vars. higher #prio means it changes first.
  int8_t _t0_adj_prio;
  int8_t _dt_adj_prio;
  int8_t _tmax_adj_prio;
  int8_t _nstep_adj_prio;
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

  specfem::mesh::mesh _mesh;
  specfem::quadrature::quadratures _quadratures;
  std::vector<std::shared_ptr<specfem::sources::source> > _sources;
  std::vector<std::shared_ptr<specfem::receivers::receiver> > _receivers;
  std::vector<specfem::enums::seismogram::type> _seismogram_types;
  int _nseismogram_steps; // TODO this is max_sig_step; verify that this
                          // actually is nseismo_steps

  bool needs_mesh_update;
};

} // namespace demo_assembly
} // namespace _util
#endif
