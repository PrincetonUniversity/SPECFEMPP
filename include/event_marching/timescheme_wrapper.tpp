#ifndef _SPECFEM_EVENT_MARCHING_TIMESCHEME_WRAPPER_TPP
#define _SPECFEM_EVENT_MARCHING_TIMESCHEME_WRAPPER_TPP

#include "timescheme_wrapper.hpp"




namespace specfem {
namespace event_marching {

template <typename TimeScheme>
void timescheme_wrapper<TimeScheme>::register_under_marcher(specfem::event_marching::event_marcher* marcher){
  //if not already registered, register it
  if(registrations.insert(marcher).second){
    marcher->register_event(&step_event);
  }
}
template <typename TimeScheme>
void timescheme_wrapper<TimeScheme>::unregister_from_marcher(specfem::event_marching::event_marcher* marcher){
  auto it = registrations.find(marcher);
  if(it != registrations.end()){
    marcher->unregister_event(&step_event);
    registrations.erase(it);
  }
}

template <typename TimeScheme>
void timescheme_wrapper<TimeScheme>::set_forward_predictor_event(
      specfem::element::medium_tag medium, precedence p){
  time_stepper.unregister_event(forward_predictor_events[medium].get());

  forward_predictor_events[medium] = std::make_unique<specfem::event_marching::forward_predictor_event<TimeScheme>>(*this,medium,p);
  time_stepper.register_event(forward_predictor_events[medium].get());

}

template <typename TimeScheme>
void timescheme_wrapper<TimeScheme>::set_forward_corrector_event(
      specfem::element::medium_tag medium, precedence p){
  time_stepper.unregister_event(forward_corrector_events[medium].get());

  forward_corrector_events[medium] = std::make_unique<specfem::event_marching::forward_corrector_event<TimeScheme>>(*this,medium,p);
  time_stepper.register_event(forward_corrector_events[medium].get());

}

template <typename TimeScheme>
template <specfem::element::medium_tag medium, specfem::wavefield::simulation_field WaveFieldType,
        specfem::dimension::type DimensionType, typename qp_type>
void timescheme_wrapper<TimeScheme>::set_wavefield_update_event(
      specfem::kernels::kernels<WaveFieldType, DimensionType, qp_type> &kernels,
      precedence p){
  time_stepper.unregister_event(wavefield_update_events[medium].get());

  wavefield_update_events[medium] = std::make_unique<
      specfem::event_marching::wavefield_update_event<TimeScheme,medium,WaveFieldType,DimensionType,qp_type>>
      (*this,kernels,p);
  time_stepper.register_event(wavefield_update_events[medium].get());

}

template <typename TimeScheme>
template <specfem::wavefield::simulation_field WaveFieldType,
        specfem::dimension::type DimensionType, typename qp_type>
void timescheme_wrapper<TimeScheme>::set_seismogram_update_event(
      specfem::kernels::kernels<WaveFieldType, DimensionType, qp_type> &kernels, precedence p){
  time_stepper.unregister_event(seismogram_update_events[WaveFieldType].get());

  seismogram_update_events[WaveFieldType] = std::make_unique<
      specfem::event_marching::seismogram_update_event<TimeScheme,WaveFieldType,DimensionType,qp_type>>
      (*this,kernels,p);

  time_stepper.register_event(seismogram_update_events[WaveFieldType].get());

}


template <typename TimeScheme>
void timescheme_wrapper<TimeScheme>::set_plotter_update_event(
      std::vector<std::shared_ptr<specfem::plotter::plotter> > &plotters,
                                   precedence p){
  if(plotter_update_event != nullptr)
    time_stepper.unregister_event(plotter_update_event.get());
  plotter_update_event = std::make_unique<specfem::event_marching::plotter_update_event<TimeScheme>>(*this,plotters,p);
  time_stepper.register_event(plotter_update_event.get());
}



} // namespace event_marching
} // namespace specfem

#endif
