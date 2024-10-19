#ifndef _SPECFEM_EVENT_MARCHING_EVENT_HPP
#define _SPECFEM_EVENT_MARCHING_EVENT_HPP


#include "event_marching/interface.hpp"

namespace specfem {
namespace event_marching {


class event{
  public:
    event(specfem::event_marching::precedence p, specfem::event_marching::event_call c):
      precedence(p), event_call(c){}

  private:
    specfem::event_marching::precedence precedence;
    specfem::event_marching::event_call event_call;
};


}
}

#endif