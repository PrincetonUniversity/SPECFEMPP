#ifndef _FRECHET_DERIVATIVES_FRECHLET_DERIVATIVES_HPP
#define _FRECHET_DERIVATIVES_FRECHLET_DERIVATIVES_HPP

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "impl/frechet_element.hpp"
#include "impl/frechet_element.tpp"

namespace specfem {
namespace frechet_derivatives {
template <int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
class frechet_derivatives {
public:
  using dimension = specfem::dimension::dimension<DimensionType>;
  using medium_type = specfem::medium::medium<DimensionType, MediumTag>;

  frechet_derivatives(const specfem::compute::assembly &assembly)
      : isotropic_elements(assembly) {}

  void compute(const type_real &dt) { isotropic_elements.compute(dt); }

private:
  specfem::frechet_derivatives::impl::frechet_elements<
      NGLL, DimensionType, MediumTag, specfem::element::property_tag::isotropic>
      isotropic_elements;
};
} // namespace frechet_derivatives
} // namespace specfem

#endif /* _FRECHET_DERIVATIVES_FRECHLET_DERIVATIVES_HPP */
