#ifndef _SPECFEM_EVENT_MARCHING_EVENT_HPP
#define _SPECFEM_EVENT_MARCHING_EVENT_HPP

#include "event_marcher.hpp"
#include <functional>

namespace specfem {
namespace event_marching {

class event {
public:
  // static bool precedence_comp(const event& a, const event& b) {
  //   return a.precedence < b.precedence;
  // }
  static bool precedence_comp(const event *a, const event *b) {
    return a->precedence < b->precedence;
  }

  event(specfem::event_marching::precedence p) : precedence(p) {}

  event() : precedence(specfem::event_marching::DEFAULT_EVENT_PRECEDENCE) {}

  virtual int call() { return specfem::event_marching::DEFAULT_EVENT_CALL(); }
  const specfem::event_marching::precedence get_precedence() {
    return precedence;
  }

private:
  specfem::event_marching::precedence precedence;
};

class arbitrary_call_event : public event {
public:
  arbitrary_call_event(std::function<int()> event_call_func,
                       specfem::event_marching::precedence p)
      : event_call_func(event_call_func), event(p) {}

  int call() { return event_call_func(); }

private:
  std::function<int()> event_call_func;
};

} // namespace event_marching
} // namespace specfem

#endif
