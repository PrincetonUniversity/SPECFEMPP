#ifndef _SPECFEM_EVENT_MARCHING_EVENT_HPP
#define _SPECFEM_EVENT_MARCHING_EVENT_HPP


#include "event_marcher.hpp"

namespace specfem {
namespace event_marching {


class event{
  public:
    event(specfem::event_marching::precedence p, specfem::event_marching::event_call c):
      precedence(p), event_call(c){}

    event():
      precedence(specfem::event_marching::DEFAULT_EVENT_PRECEDENCE),
      event_call(specfem::event_marching::DEFAULT_EVENT_CALL){}

  private:
    const specfem::event_marching::precedence precedence;
    const specfem::event_marching::event_call event_call;
};


}
}

#endif