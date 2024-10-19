#ifndef _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_HPP
#define _SPECFEM_EVENT_MARCHING_EVENT_MARCHER_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/simulation.hpp"
#include "kernels/frechet_kernels.hpp"
#include "kernels/kernels.hpp"
#include "solver/solver.hpp"

#include "event_marching/interface.hpp"

namespace specfem {
namespace event_marching {

template <specfem::simulation::type Simulation,
          specfem::dimension::type DimensionType, typename qp_type>
class event_marcher;



template <specfem::dimension::type DimensionType, typename qp_type>
class event_marcher<specfem::simulation::type::forward, DimensionType, qp_type>
    : public specfem::solver::solver {
public:
  event_marcher(
      const specfem::kernels::kernels<specfem::wavefield::type::forward,
                                      DimensionType, qp_type> &kernels)
      : kernels(kernels) {}

  void run();

private:
  specfem::kernels::kernels<specfem::wavefield::type::forward, DimensionType,
                            qp_type>
      kernels;
  
  //these are to be called without any invokers/interrupts.
  std::vector<specfem::event_marching::event> main_events;
};


} // namespace event_marching
} // namespace specfem


#include "event_marching/event_marcher.tpp"
#endif
