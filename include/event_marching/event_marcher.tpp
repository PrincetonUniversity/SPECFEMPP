#ifndef _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_TPP
#define _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_TPP

#include "event_marching/event_marcher.hpp"

#include <algorithm>

namespace specfem{
namespace event_marching{


void event_marcher::register_main_event(specfem::event_marching::event event){
  queued_registration.push_back(std::make_pair(event,true));
}
void event_marcher::unregister_main_event(specfem::event_marching::event event){
  queued_registration.push_back(std::make_pair(event,false));
}
void event_marcher::process_registrations(){
  for(const auto [event, reg] : queued_registration){
    if (reg){
      main_events.push_back(event);
    }else{
      // can't do this since event doesnt have a ==
      //std::remove(main_events.begin(),main_events.end(),event);
      throw std::runtime_error("Attempt to remove an event!");
    }
    
  }
  std::sort(main_events.begin(),main_events.end(),specfem::event_marching::event::precedence_comp);
}

void event_marcher::run(){
  process_registrations();
  for( const auto event : main_events){
    //todo handle int return

    std::cout << "event @ "<< event.precedence <<": " << event.event_call() << "\n";
  }
}




}
}
#endif