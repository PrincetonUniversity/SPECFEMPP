#ifndef _SPECFEM_SOLVER_TIME_MARCHING_HPP
#define _SPECFEM_SOLVER_TIME_MARCHING_HPP

#include "coupled_interface/interface.hpp"
#include "domain/interface.hpp"
#include "enumerations/interface.hpp"
#include "solver.hpp"
#include "timescheme/interface.hpp"

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
class time_marching<specfem::simulation::type::adjoint, DimensionType, qp_type>
    : public solver {
public:
  time_marching(
      const specfem::kernels::kernels<specfem::wavefield::type::adjoint,
                                      DimensionType, qp_type> &adjoint_kernels,
      const specfem::kernels::kernels<specfem::wavefield::type::backward,
                                      DimensionType, qp_type> &backward_kernels,
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme)
      : adjoint_kernels(adjoint_kernels), backward_kernels(backward_kernels),
        time_scheme(time_scheme) {}

  void run() override;

private:
  specfem::kernels::kernels<specfem::wavefield::type::adjoint, DimensionType,
                            qp_type>
      adjoint_kernels;
  specfem::kernels::kernels<specfem::wavefield::type::backward, DimensionType,
                            qp_type>
      backward_kernels;
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme;
};
} // namespace solver
} // namespace specfem

#endif
