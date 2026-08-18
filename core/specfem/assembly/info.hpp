#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/info/impl/bounding_box.hpp"
#include "specfem/assembly/info/impl/bounds.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/element.hpp"
#include "specfem/setup.hpp"

namespace specfem::assembly {

/**
 * @brief Computes and stores mesh statistics and numerical stability
 * parameters.
 *
 * Analyzes the assembled mesh to extract spatial bounds, material property
 * ranges, and element geometry statistics. Estimates the minimum resolvable
 * period and suggests a time step based on the CFL condition.
 *
 * The minimum period estimation follows Komatitsch et al. (2005):
 * "average number of points per minimum wavelength in an element should be
 * around 5."
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 *
 * @note The minimum period is an empirical estimate, not a sharp cutoff.
 *       Synthetics become progressively less accurate for shorter periods.
 */
template <specfem::element::dimension_tag DimensionTag> struct Info {
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag

  Info() = default;

  /**
   * @brief Construct mesh info by analyzing mesh geometry and material
   * properties.
   *
   * @param mesh Assembled mesh containing element geometry and GLL points
   * @param properties Material properties at all mesh points
   * @param element_types Element classification by medium and property type
   */
  Info(const specfem::assembly::mesh<dimension_tag> &mesh,
       const specfem::assembly::properties<dimension_tag> &properties,
       const specfem::assembly::element_types<dimension_tag> &element_types);

  info::impl::BoundingBox<dimension_tag> domain_bounds; ///< Spatial extent of
                                                        ///< the mesh domain
  info::impl::Bounds element_size; ///< Element corner-to-corner distances
  info::impl::Bounds gll_distance; ///< Distances between adjacent GLL points
  info::impl::Bounds vp;           ///< P-wave velocity range
  info::impl::Bounds vs;           ///< S-wave velocity range
  info::impl::Bounds v;            ///< Combined wave velocity range
  info::impl::Bounds rho;          ///< Density range
  info::impl::Bounds vp_vs_ratio;  ///< Vp/Vs ratio range

  type_real suggested_time_step;    ///< Time step satisfying CFL condition
  type_real largest_minimum_period; ///< Maximum of minimum resolvable periods
                                    ///< across elements

  /**
   * @brief Generate formatted string representation of mesh statistics.
   * @return Multi-line string with labeled mesh properties
   */
  std::string string() const;
};

} // namespace specfem::assembly

extern template specfem::assembly::Info<specfem::element::dimension_tag::dim2>::
    Info(const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &,
         const specfem::assembly::properties<
             specfem::element::dimension_tag::dim2> &,
         const specfem::assembly::element_types<
             specfem::element::dimension_tag::dim2> &);
extern template specfem::assembly::Info<specfem::element::dimension_tag::dim3>::
    Info(const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
         const specfem::assembly::properties<
             specfem::element::dimension_tag::dim3> &,
         const specfem::assembly::element_types<
             specfem::element::dimension_tag::dim3> &);
