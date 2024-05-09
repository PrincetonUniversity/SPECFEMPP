#ifndef _SPECFEM_KERNELS_FRECHET_KERNELS_HPP
#define _SPECFEM_KERNELS_FRECHET_KERNELS_HPP

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "frechet_derivatives/frechet_derivatives.hpp"

namespace specfem {
namespace kernels {

template <int NGLL, specfem::dimension::type DimensionType>
class frechet_kernels {
public:
  frechet_kernels(const specfem::compute::assembly &assembly)
      : elastic_elements(assembly), acoustic_elements(assembly) {}

  inline void compute_derivatives(const type_real &dt) {
    elastic_elements.compute(dt);
    acoustic_elements.compute(dt);
  }

private:
  specfem::frechet_derivatives::frechet_derivatives<
      NGLL, DimensionType, specfem::element::medium_tag::elastic>
      elastic_elements;

  specfem::frechet_derivatives::frechet_derivatives<
      NGLL, DimensionType, specfem::element::medium_tag::acoustic>
      acoustic_elements;
};

} // namespace kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_FRECHET_KERNELS_HPP */
