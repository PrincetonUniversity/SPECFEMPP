#ifndef _SPECFEM_KERNELS_HPP
#define _SPECFEM_KERNELS_HPP

#include "compute/interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/simulation.hpp"
#include "impl/domain_kernels.hpp"

namespace specfem {
namespace kernels {
template <specfem::simulation::type simulation_type,
          specfem::dimension::type DimensionType, typename qp_type>
class kernels;

template <specfem::dimension::type DimensionType, typename qp_type>
class kernels<specfem::simulation::type::forward, DimensionType, qp_type>
    : public impl::domain_kernels<specfem::simulation::type::forward,
                                  DimensionType, qp_type> {
public:
  kernels(const specfem::compute::assembly &assembly,
          const qp_type &quadrature_points)
      : impl::domain_kernels<specfem::simulation::type::forward, DimensionType,
                             qp_type>(assembly, quadrature_points) {}
};

// template <typename qp_type>
// class kernels<specfem::enums::simulation::type::adjoint, qp_type>
//     : public impl::domain_kernels<qp_type,
//                                   specfem::enums::simulation::type::adjoint>,
//       public impl::frechlet_kernels<qp_type> {
//   kernels(const specfem::compute::assembly &assembly,
//           const qp_type &quadrature_points)
//       : impl::domain_kernels<qp_type,
//                              specfem::enums::simulation::type::adjoint>(
//             assembly, quadrature_points),
//         impl::frechlet_kernels<qp_type>(assembly, quadrature_points) {}
// };

} // namespace kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_HPP */
