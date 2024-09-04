#ifndef _SPECFEM_SOLVER_TIME_MARCHING_HPP
#define _SPECFEM_SOLVER_TIME_MARCHING_HPP

#include "coupled_interface/interface.hpp"
#include "domain/domain.hpp"
#include "enumerations/interface.hpp"
#include "kernels/frechet_kernels.hpp"
#include "kernels/kernels.hpp"
#include "solver.hpp"
#include "timescheme/newmark.hpp"

namespace specfem {
namespace solver {
template <specfem::simulation::type Simulation,
          specfem::dimension::type DimensionType, typename qp_type>
class time_marching;

template <specfem::dimension::type DimensionType, typename qp_type>
class time_marching<specfem::simulation::type::forward, DimensionType, qp_type>
    : public solver {
public:
  time_marching(
      const specfem::kernels::kernels<specfem::wavefield::type::forward,
                                      DimensionType, qp_type> &kernels,
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme)
      : kernels(kernels), time_scheme(time_scheme) {}

  void run() override;

private:
  specfem::kernels::kernels<specfem::wavefield::type::forward, DimensionType,
                            qp_type>
      kernels;
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme;
};

template <specfem::dimension::type DimensionType, typename qp_type>
class time_marching<specfem::simulation::type::combined, DimensionType, qp_type>
    : public solver {
public:
  time_marching(
      const specfem::compute::assembly &assembly,
      const specfem::kernels::kernels<specfem::wavefield::type::adjoint,
                                      DimensionType, qp_type> &adjoint_kernels,
      const specfem::kernels::kernels<specfem::wavefield::type::backward,
                                      DimensionType, qp_type> &backward_kernels,
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme)
      : assembly(assembly), adjoint_kernels(adjoint_kernels),
        frechet_kernels(assembly), backward_kernels(backward_kernels),
        time_scheme(time_scheme) {}

  void run() override;

private:
  constexpr static int NGLL = qp_type::NGLL;
  specfem::kernels::kernels<specfem::wavefield::type::adjoint, DimensionType,
                            qp_type>
      adjoint_kernels;
  specfem::kernels::kernels<specfem::wavefield::type::backward, DimensionType,
                            qp_type>
      backward_kernels;
  specfem::kernels::frechet_kernels<DimensionType, NGLL> frechet_kernels;
  specfem::compute::assembly assembly;
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme;
};
} // namespace solver
} // namespace specfem

#endif
