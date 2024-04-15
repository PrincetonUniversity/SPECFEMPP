#ifndef _DOMAIN_HPP
#define _DOMAIN_HPP

#include "compute/interface.hpp"
#include "impl/interface.hpp"
#include "impl/kernels.hpp"
#include "impl/receivers/interface.hpp"
#include "impl/sources/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename qp_type>
class domain : public specfem::domain::impl::kernels::kernels<
                   WavefieldType, DimensionType, MediumTag, qp_type> {
public:
  using dimension = specfem::dimension::dimension<DimensionType>;
  using medium_type = specfem::medium::medium<DimensionType, MediumTag>;
  using quadrature_points_type = qp_type; ///< Type of quadrature points i.e.
                                          ///< static or dynamic

  domain(const specfem::compute::assembly &assembly,
         const quadrature_points_type &quadrature_points)
      : field(assembly.fields.get_simulation_field<WavefieldType>()
                  .template get_field<MediumTag>()),
        specfem::domain::impl::kernels::kernels<
            WavefieldType, DimensionType, MediumTag, quadrature_points_type>(
            assembly, quadrature_points) {}

  ~domain() = default;

  void invert_mass_matrix();

  void divide_mass_matrix();

private:
  specfem::compute::impl::field_impl<DimensionType, MediumTag>
      field; ///< Field object
};
} // namespace domain

} // namespace specfem

#endif
