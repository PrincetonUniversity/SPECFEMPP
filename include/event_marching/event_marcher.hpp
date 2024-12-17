#ifndef _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_HPP
#define _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/simulation.hpp"
#include "kernels/frechet_kernels.hpp"
#include "kernels/kernels.hpp"
#include "solver/solver.hpp"

#include <cmath>
#include <forward_list>
#include <functional>
#include <utility>
#include <vector>

// event_marching namespace expects these to exist:
namespace specfem {
namespace event_marching {

class event_marcher;

// dataype we use to store the event order
// (ideally would be low-precision float but efficiency is not mission critical)
typedef float precedence;

// style for event callbacks (if this is good TBD):
// takes invoker (event_marcher that calls the event), but returns a success
// state
// typedef int (*event_call)();
typedef std::function<int()> event_call;

// event defaults.
constexpr precedence DEFAULT_EVENT_PRECEDENCE = 0;
constexpr precedence PRECEDENCE_BEFORE_INIT = -INFINITY;
constexpr precedence PRECEDENCE_AFTER_END = INFINITY;
int _DEFAULT_EVENT_CALL() { return 0; }
const event_call DEFAULT_EVENT_CALL = []() { return 0; };
} // namespace event_marching
} // namespace specfem

#include "event.hpp"

namespace specfem {
namespace event_marching {

class event_marcher {
public:
  event_marcher()
      : _current_step(events.before_begin()),
        _current_precedence(specfem::event_marching::PRECEDENCE_BEFORE_INIT) {}

  int march_events();
  int march_events(int num_steps);

  void register_event(specfem::event_marching::event *event);
  void unregister_event(specfem::event_marching::event *event);

  precedence current_precedence();
  void set_current_precedence(precedence p);
  void goto_beginning() {
    set_current_precedence(specfem::event_marching::PRECEDENCE_BEFORE_INIT);
  }

protected:
  // these are to be called without any invokers/interrupts.
  std::forward_list<specfem::event_marching::event *> events;

  //<event to manage, true: add / false: remove>
  std::vector<std::pair<specfem::event_marching::event *, bool> >
      queued_event_registration;
  void process_registrations();

  std::forward_list<specfem::event_marching::event *>::iterator current_step();

private:
  std::forward_list<specfem::event_marching::event *>::iterator _current_step;
  precedence _current_precedence;
};

class event_system : public specfem::solver::solver,
                     public specfem::event_marching::event_marcher {
public:
  event_system() {}

  void run();
};

} // namespace event_marching
} // namespace specfem

#include "event_marching/event_marcher.tpp"
#endif
