#pragma once
#include "enumerations/interface.hpp"
#include "impl/bounding_box.hpp"


namespace specfem::assembly {

template <specfem::dimension::type DimensionTag> struct assembly;

/*
! estimation of minimum period resolved
! based on average GLL distance within element and minimum velocity
!
! rule of thumb (Komatitsch et al. 2005):
! "average number of points per minimum wavelength in an element should be around 5."

! average distance between GLL points within this element
avg_distance = elemsize_max / ( NGLLX - 1)  ! since NGLLX = NGLLY = NGLLZ

! largest possible minimum period such that number of points per minimum wavelength
! npts = ( min(vpmin,vsmin)  * pmax ) / avg_distance  is about ~ NPTS_PER_WAVELENGTH
!
! note: obviously, this estimation depends on the choice of points per wavelength
!          which is empirical at the moment.
!          also, keep in mind that the minimum period is just an estimation and
!          there is no such sharp cut-off period for valid synthetics.
!          seismograms become just more and more inaccurate for periods shorter than this estimate.
*/

template <>
struct Info<specfem::dimension::type::dim2> {
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag

  Info() = default;

  Info(specfem::assembly::assembly<dimension_tag> &assembly);

  // Computed mesh properties
  info::impl::BoundingBox<dimension_tag> domain_bounds;
  info::impl::Bounds element_size;
  info::impl::Bounds gll_distance;
  info::impl::Bounds jacobian_determinant;
  info::impl::Bounds vp;
  info::impl::Bounds vs;
  info::impl::Bounds v;
  info::impl::Bounds rho;
  info::impl::Bounds vp_vs_ratio;

  // Suggested time step based on CFL condition
  type_real suggested_time_step;
  type_real largest_minimum_period;


  std::string string() const;
};


} // namespace specfem::assembly