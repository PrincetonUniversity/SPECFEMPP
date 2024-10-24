#ifndef _SPECFEM_EVENT_MARCHING_TIMEMARCHING_WRAPPER_TPP
#define _SPECFEM_EVENT_MARCHING_TIMEMARCHING_WRAPPER_TPP

#include "timemarching_wrapper.hpp"




namespace specfem {
namespace event_marching {


// template <specfem::dimension::type DimensionType, typename qp_type>
// timemarching_wrapper<specfem::simulation::type::forward, DimensionType, qp_type>::timemarching_wrapper(
//         specfem::solver::time_marching<specfem::simulation::type::forward, DimensionType, qp_type> time_marcher
//   ): time_marcher(time_marcher), registration(nullptr){}

template <specfem::dimension::type DimensionType, typename qp_type>
void timemarching_wrapper<specfem::simulation::type::forward, DimensionType, qp_type>
    ::load_into_marcher_main_events(
      specfem::event_marching::event_marcher& marcher
    ){
  if(registration != nullptr){
    throw std::runtime_error(
"timemarching_wrapper: Attempting to register a marcher when the wrapper is already registered!"
    );
  }
  registration = &marcher;

  //register predictors (every media)
  for(const std::pair<const specfem::element::medium_tag,
      specfem::event_marching::event>& entry : forward_predictor_events){
    marcher.register_main_event(entry.second);
  }
  //register correctors (every media)
  for(const std::pair<const specfem::element::medium_tag,
      specfem::event_marching::event>& entry : forward_corrector_events){
    marcher.register_main_event(entry.second);
  }
}

  

template <specfem::dimension::type DimensionType, typename qp_type>
void timemarching_wrapper<specfem::simulation::type::forward, DimensionType, qp_type>
    ::unload_from_marcher_main_events(){
  if(registration == nullptr){
    throw std::runtime_error(
"timemarching_wrapper: Attempting to unregister an unregistered marcher!"
    );
  }

  //unregister predictors (every media)
  for(const std::pair<const specfem::element::medium_tag,
      specfem::event_marching::event>& entry : forward_predictor_events){
    registration->unregister_main_event(entry.second);
  }
  //unregister correctors (every media)
  for(const std::pair<const specfem::element::medium_tag,
      specfem::event_marching::event>& entry : forward_corrector_events){
    registration->unregister_main_event(entry.second);
  }
  registration = nullptr;
}


template <specfem::dimension::type DimensionType, typename qp_type>
void timemarching_wrapper<specfem::simulation::type::forward, DimensionType, qp_type>
    ::set_forward_predictor_precedence(specfem::element::medium_tag medium, precedence p){
}

template <specfem::dimension::type DimensionType, typename qp_type>
void timemarching_wrapper<specfem::simulation::type::forward, DimensionType, qp_type>
    ::set_forward_corrector_precedence(specfem::element::medium_tag medium, precedence p){
}



} // namespace event_marching
} // namespace specfem

#endif
