#pragma once
#include "specfem/assembly.hpp"
#include "utilities/include/fieldmanip/fieldsetter.hpp"

namespace specfem::nonconforming_test::kernel {

void test_nonconforming_acoustic_elastic(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const std::string &meshname);

void test_nonconforming_acoustic_elastic(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname);

void test_nonconforming_acoustic_elastic_conjugate(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname);

/**
 * @brief Sets the acoustic acceleration field to power function.
 *
 * when passed to specfem::test_fieldmanip::set_field_values(), sets
 * acceleration to (x/xscale)^{xpow} (y/yscale)^{ypow}. Zeroes out displacement
 * and velocity.
 */
template <int xpow, int ypow>
struct acoustic_field_initializer_pow
    : public specfem::test_fieldmanip::PointSetter<
          specfem::element::dimension_tag::dim3,
          specfem::element::medium_tag::acoustic> {
  type_real inv_xscale;
  type_real inv_yscale;
  KOKKOS_INLINE_FUNCTION PointAccelerationType
  acceleration(const PointData &data) const {
    PointAccelerationType val;
    constexpr int ncomp =
        specfem::element::attributes<dimension_tag, medium_tag>::components;
    for (int icomp = 0; icomp < ncomp; icomp++) {
      val(icomp) = std::pow(data.coords.x * inv_xscale, xpow) *
                   std::pow(data.coords.y * inv_yscale, ypow);
    }
    return val;
  }
  acoustic_field_initializer_pow(const type_real &xscale,
                                 const type_real &yscale)
      : PointSetter(true, true, true), inv_xscale(1 / xscale),
        inv_yscale(1 / yscale) {};
};

/**
 * @brief Sets the elastic displacement field to power.
 *
 * when passed to specfem::test_fieldmanip::set_field_values(), sets
 * displacement to dir * (x/xscale)^{xpow} (y/yscale)^{ypow}, where dir is a
 * constant vector. Zeroes out displacement and velocity.
 */
template <int xpow, int ypow>
struct elastic_field_initializer_pow
    : public specfem::test_fieldmanip::PointSetter<
          specfem::element::dimension_tag::dim3,
          specfem::element::medium_tag::elastic> {
  type_real setdir[ndim];
  type_real inv_xscale;
  type_real inv_yscale;

  KOKKOS_INLINE_FUNCTION PointDisplacementType
  displacement(const PointData &data) const {
    PointDisplacementType val;
    constexpr int ncomp =
        specfem::element::attributes<dimension_tag, medium_tag>::components;
    type_real powpos = std::pow(data.coords.x * inv_xscale, xpow) *
                       std::pow(data.coords.y * inv_yscale, ypow);

    for (int icomp = 0; icomp < ncomp; icomp++) {
      val(icomp) = powpos * setdir[icomp];
    }
    return val;
  }
  elastic_field_initializer_pow(const type_real &xscale,
                                const type_real &yscale)
      : PointSetter(true, true, true), setdir{ 0, 0, 1 },
        inv_xscale(1 / xscale), inv_yscale(1 / yscale) {};
};

} // namespace specfem::nonconforming_test::kernel
