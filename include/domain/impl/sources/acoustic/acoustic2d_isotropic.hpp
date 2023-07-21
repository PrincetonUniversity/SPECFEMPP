#ifndef _DOMAIN_SOURCE_ACOUSTIC_ISOTROPIC2D_HPP
#define _DOMAIN_SOURCE_ACOUSTIC_ISOTROPIC2D_HPP

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/acoustic2d.hpp"
#include "domain/impl/sources/source.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

/**
 * @brief Decltype for the field subviewed at particular global index
 *
 */
using field_type = Kokkos::Subview<
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
    std::remove_const_t<decltype(Kokkos::ALL)> >;

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
/**
 * @brief Elemenatal source class for 2D isotropic acoustic medium with number
 * of quadrature points defined at compile time
 *
 * @tparam N Number of Gauss-Lobatto-Legendre quadrature points
 */
template <int N>
class source<specfem::enums::element::dimension::dim2,
             specfem::enums::element::medium::acoustic,
             specfem::enums::element::quadrature::static_quadrature_points<N>,
             specfem::enums::element::property::isotropic>
    : public source<
          specfem::enums::element::dimension::dim2,
          specfem::enums::element::medium::acoustic,
          specfem::enums::element::quadrature::static_quadrature_points<N> > {

public:
  /**
   * @brief Default elemental source constructor
   *
   */
  KOKKOS_FUNCTION source() = default;

  /**
   * @brief Default elemental source copy constructor
   *
   */
  KOKKOS_FUNCTION source(const source &) = default;

  /**
   * @brief Construct a new elemental source object
   *
   * @param ispec Index of the element where the source is located
   * @param kappa Kappa array
   * @param source_array Source array containing pre-computed lagrange
   * interpolants
   * @param stf Pointer to the source time function object
   */
  KOKKOS_FUNCTION source(const int &ispec,
                         const specfem::kokkos::DeviceView2d<type_real> &kappa,
                         specfem::kokkos::DeviceView3d<type_real> source_array,
                         specfem::forcing_function::stf *stf);

  /**
   * @brief Compute the interaction of the source with the medium computed at
   * the quadrature point xz
   *
   * @param xz Quadrature point index in the element
   * @param stf_value Value of the source time function at the current time step
   * @param accel Acceleration in the x direction at the quadrature point
   * (return value)
   */
  KOKKOS_INLINE_FUNCTION void
  compute_interaction(const int &xz, const type_real &stf_value,
                      type_real *accel) const override;

  /**
   * @brief Compute the value of the source time function at time t
   *
   * @param t Time
   * @return type_real Value of the source time function at time t
   */
  KOKKOS_INLINE_FUNCTION type_real eval_stf(const type_real &t) const override {
    return stf->compute(t);
  }

  /**
   * @brief Update the acceleration at the quadrature point xz
   *
   * @param accel Acceleration at the quadrature point as
   * computed by compute_interaction
   * @param field_dot_dot Acceleration field subviewed at global index
   * ibool(ispec, iz, ix)
   */
  KOKKOS_INLINE_FUNCTION void
  update_acceleration(const type_real &accel,
                      field_type field_dot_dot) const override;

  /**
   * @brief Get the index of the element where the source is located
   *
   * @return int Index of the element where the source is located
   */
  KOKKOS_INLINE_FUNCTION int get_ispec() const override { return ispec; }

private:
  int ispec; ///< Index of the element where the source is located
  specfem::kokkos::DeviceView2d<type_real> kappa; /// kappa array
  specfem::forcing_function::stf *stf; ///< Pointer to the source time function
                                       ///< object
  specfem::kokkos::DeviceView3d<type_real> source_array; ///< Source array
                                                         ///< containing
                                                         ///< pre-computed
                                                         ///< lagrange
                                                         ///< interpolants
};
} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
