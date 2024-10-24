#ifndef _SPECFEM_EVENT_MARCHING_TIMEMARCHING_WRAPPER_HPP
#define _SPECFEM_EVENT_MARCHING_TIMEMARCHING_WRAPPER_HPP

#include "event_marcher.hpp"
#include "event.hpp"

#include "solver/time_marching.hpp"

#include <unordered_map>
#include <set>

namespace specfem {
namespace event_marching {


template <specfem::simulation::type Simulation,
          specfem::dimension::type DimensionType, typename qp_type>
class timemarching_wrapper{
public:
  void load_into_marcher_main_events(specfem::event_marching::event_marcher& marcher);
  void unload_from_marcher_main_events();

  specfem::solver::time_marching<Simulation, DimensionType, qp_type> time_marcher;
private:
  specfem::event_marching::event_marcher* registration;

};



template <specfem::dimension::type DimensionType, typename qp_type>
class timemarching_wrapper<specfem::simulation::type::forward, DimensionType, qp_type>{
public:

  timemarching_wrapper(specfem::solver::time_marching<specfem::simulation::type::forward,
        DimensionType, qp_type>& time_marcher
  ): registration(nullptr), time_marcher(time_marcher) {}
  void set_forward_predictor_precedence(specfem::element::medium_tag medium, precedence p);
  void set_forward_corrector_precedence(specfem::element::medium_tag medium, precedence p);

  void load_into_marcher_main_events(specfem::event_marching::event_marcher& marcher);
  void unload_from_marcher_main_events();

  specfem::solver::time_marching<specfem::simulation::type::forward, DimensionType, qp_type> time_marcher;
private:
  specfem::event_marching::event_marcher* registration;

  std::unordered_map<specfem::element::medium_tag, specfem::event_marching::event>
      forward_predictor_events;
  std::unordered_map<specfem::element::medium_tag, specfem::event_marching::event>
      forward_corrector_events;
  std::unordered_map<specfem::element::medium_tag, specfem::event_marching::event>
      update_wavefield_event;
  std::unordered_map<specfem::element::medium_tag, specfem::event_marching::event>
      seismogram_update_event;

};


} // namespace event_marching
} // namespace specfem

#include "timemarching_wrapper.tpp"
#endif
