#ifndef _SPECFEM_EVENT_MARCHING_INTERFACE_HPP
#define _SPECFEM_EVENT_MARCHING_INTERFACE_HPP



namespace specfem {
namespace event_marching {

// dataype we use to store the event order
// (ideally would be low-precision float but efficiency is not mission critical)
typedef float precedence;

// style for event callbacks (if this is good TBD):
// takes no arguments, but returns a success state
typedef int (*event_call)();



}
}

#include "event.hpp"
#include "event_marcher.hpp"

#endif