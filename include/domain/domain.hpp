#ifndef _DOMAIN_HPP
#define _DOMAIN_HPP

#include "compute/interface.hpp"
#include "impl/interface.hpp"
#include "impl/receivers/interface.hpp"
#include "impl/sources/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {

template <specfem::enums::simulation::type simulation, typename medium,
          typename qp_type>
class domain
    : public specfem::domain::impl::kernels<simulation, medium, qp_type> {
public:
  using dimension = specfem::enums::element::dimension::dim2; ///< Dimension of
                                                              ///< the domain
  using medium_type = medium; ///< Type of medium i.e. acoustic, elastic or
                              ///< poroelastic
  using quadrature_points_type = qp_type; ///< Type of quadrature points i.e.
                                          ///< static or dynamic

  domain(const specfem::compute::assembly &assembly,
         const quadrature_points_type &quadrature_points)
      : field(assembly.fields.get_simulation_field<simulation>()
                  .get_field<medium>()),
        specfem::domain::impl::kernels<simulation, medium,
                                       quadrature_points_type>(
            assembly, quadrature_points) {}

  ~domain() = default;

  void invert_mass_matrix();

  void divide_mass_matrix();

private:
  specfem::compute::impl::field_impl<medium_type> field; ///< Field object
};
} // namespace domain

} // namespace specfem

#endif
