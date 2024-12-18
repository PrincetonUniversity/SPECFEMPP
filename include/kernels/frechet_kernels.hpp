#ifndef _SPECFEM_KERNELS_FRECHET_KERNELS_HPP
#define _SPECFEM_KERNELS_FRECHET_KERNELS_HPP

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "frechet_derivatives/frechet_derivatives.hpp"

namespace specfem {
namespace kernels {

template <specfem::dimension::type DimensionType, int NGLL>
class frechet_kernels {
public:
  constexpr static auto dimension = DimensionType;
  constexpr static auto ngll = NGLL;
  frechet_kernels(const specfem::compute::assembly &assembly)
      : assembly(assembly) {}

  inline void compute_derivatives(const type_real &dt) {
#define CALL_COMPUTE_MATERIAL_DERIVATIVES(DIMENSION_TAG, MEDIUM_TAG,           \
                                          PROPERTY_TAG)                        \
  if constexpr (dimension == DIMENSION_TAG) {                                  \
    compute_material_derivatives<MEDIUM_TAG, PROPERTY_TAG>(dt);                \
  }

    CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
        CALL_COMPUTE_MATERIAL_DERIVATIVES,
        WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef CALL_COMPUTE_MATERIAL_DERIVATIVES
  }

private:
  specfem::compute::assembly assembly;

  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  void compute_material_derivatives(const type_real &dt);
};

// template <specfem::dimension::type DimensionType, int NGLL>
// class frechet_kernels {
// public:
//   frechet_kernels(const specfem::compute::assembly &assembly)
//       : elastic_elements(assembly), acoustic_elements(assembly) {}

//   inline void compute_derivatives(const type_real &dt) {
//     elastic_elements.compute(dt);
//     acoustic_elements.compute(dt);
//   }

// private:
//   specfem::frechet_derivatives::frechet_derivatives<
//       DimensionType, specfem::element::medium_tag::elastic, NGLL>
//       elastic_elements;

//   specfem::frechet_derivatives::frechet_derivatives<
//       DimensionType, specfem::element::medium_tag::acoustic, NGLL>
//       acoustic_elements;
// };

} // namespace kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_FRECHET_KERNELS_HPP */
