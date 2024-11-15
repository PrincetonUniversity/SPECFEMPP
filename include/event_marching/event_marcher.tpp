#ifndef _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_TPP
#define _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_TPP

#include "event_marching/event_marcher.hpp"

#include <algorithm>
#include <iterator>

namespace specfem{
namespace event_marching{

void event_marcher::register_event(specfem::event_marching::event* event){
  queued_event_registration.push_back(std::make_pair(event,true));
}
void event_marcher::unregister_event(specfem::event_marching::event* event){
  queued_event_registration.push_back(std::make_pair(event,false));
}
void event_marcher::process_registrations(){
  for(const auto [event, reg] : queued_event_registration){
    if(event == nullptr){
      continue;
    }
    precedence p = event->get_precedence();
    auto it = events.begin();
    auto it_prev = events.before_begin();
    while(it != events.end() && (*it)->get_precedence() < p){
      it_prev = it;
      it++;
    }
    if (reg){ // register
      events.insert_after(it_prev,event);
    }else if(it != events.end()){ // unregister
      if(it == _current_step){
        _current_step++;
        if(_current_step != events.end()){
          _current_precedence = (*_current_step)->get_precedence();
        }
      }
      events.erase_after(it_prev);
    }
  }
  queued_event_registration.clear();


}

precedence event_marcher::current_precedence(){
  current_step();
  if(_current_step == events.before_begin()){
    return specfem::event_marching::PRECEDENCE_BEFORE_INIT;
  }
  if(_current_step == events.end()){
    return specfem::event_marching::PRECEDENCE_AFTER_END;
  }
  return _current_precedence;
}
void event_marcher::set_current_precedence(precedence p){
  _current_precedence = p;
  if(_current_step == events.end() || _current_precedence < (*_current_step)->get_precedence()){
    _current_step = events.begin();
  }
}

// #define EVENT_MARCHER_VERBOSE

std::forward_list<specfem::event_marching::event*>::iterator event_marcher::current_step(){
#ifdef EVENT_MARCHER_VERBOSE
  std::cout << "(current_step = " << _current_precedence;
#endif
  process_registrations();
  if(_current_step == events.end()){
#ifdef EVENT_MARCHER_VERBOSE
    std::cout << " @end)";
#endif
    return _current_step;
  }
  if(_current_step == events.before_begin()){
#ifdef EVENT_MARCHER_VERBOSE
    std::cout << " @before_begin -> reset iter";
#endif
    _current_step = events.begin();
  }else if(_current_precedence < (*_current_step)->get_precedence()){
    //_current_precedence was modified.
#ifdef EVENT_MARCHER_VERBOSE
    std::cout << " < "<<(*_current_step)->get_precedence()<< " ->reset iter";
#endif
    _current_step = events.begin();
  }
  while(_current_step != events.end() && _current_precedence > (*_current_step)->get_precedence()){
    _current_step++;
  }
  if(_current_step == events.end()){
    _current_precedence = specfem::event_marching::PRECEDENCE_AFTER_END;
    return _current_step;
  }
  _current_precedence = (*_current_step)->get_precedence();
#ifdef EVENT_MARCHER_VERBOSE
  std::cout << " -> "<<_current_precedence<<")";
#endif
  return _current_step;
}

int event_marcher::march_events(){
  process_registrations();
  auto it = current_step();
  if(it == events.end()){return 0;}

  while(true){
    auto event = *it;
    int call_return = event->call();
#ifdef EVENT_MARCHER_VERBOSE
    std::cout << "event @ "<< event->get_precedence() <<" (ret = " << call_return << ")";
#endif
    //event registration may have changed, so *it may be undefined; re-issue
    it = current_step();
    if(it == events.end()){
      _current_precedence = specfem::event_marching::PRECEDENCE_AFTER_END;
#ifdef EVENT_MARCHER_VERBOSE
    std::cout << " was unregistered during call; prec -> "<<_current_precedence << "\n";
#endif
      return call_return;
    }
    //this will happen most of the time; if step was removed; then current_step() would be next
    if(*it == event){
      _current_step++;
      it++;
#ifdef EVENT_MARCHER_VERBOSE
    std::cout << " standard inc";
#endif
    }
    if(it == events.end()){
      return call_return;
    }
    _current_precedence = (*_current_step)->get_precedence();
#ifdef EVENT_MARCHER_VERBOSE
    std::cout << " -> " << _current_precedence << "\n";
#endif
  }
}
int event_marcher::march_events(int num_steps){
  while(num_steps > 0){

    process_registrations();

    num_steps--;
  }
  return 0;
}

void event_system::run(){
  march_events();
}


}
}
#endif
