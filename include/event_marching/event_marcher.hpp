#ifndef _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_HPP
#define _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/simulation.hpp"
#include "kernels/frechet_kernels.hpp"
#include "kernels/kernels.hpp"
#include "solver/solver.hpp"

#include <functional>
#include <vector>
#include <utility>

//event_marching namespace expects these to exist:
namespace specfem {
namespace event_marching {


class event_marcher;

// dataype we use to store the event order
// (ideally would be low-precision float but efficiency is not mission critical)
typedef float precedence;

// style for event callbacks (if this is good TBD):
// takes invoker (event_marcher that calls the event), but returns a success state
//typedef int (*event_call)();
typedef std::function<int()> event_call;


// event defaults.
constexpr precedence DEFAULT_EVENT_PRECEDENCE = 0;
int _DEFAULT_EVENT_CALL(){return 0;}
const event_call DEFAULT_EVENT_CALL = []() {return 0;};
}
}


#include "event.hpp"

namespace specfem {
namespace event_marching {





class event_marcher: public specfem::solver::solver {
public:
  event_marcher(){}

  void run();

  void register_main_event(specfem::event_marching::event event);
  void unregister_main_event(specfem::event_marching::event event);

private:
  //these are to be called without any invokers/interrupts.
  std::vector<specfem::event_marching::event> main_events;

  //<event to manage, true: add / false: remove>
  std::vector<std::pair<specfem::event_marching::event,bool>> queued_registration;
  void process_registrations();
};


} // namespace event_marching
} // namespace specfem


#include "event_marching/event_marcher.tpp"
#endif
