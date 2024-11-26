#pragma once

#include "compute/assembly/assembly.hpp"
#include "impl/kernels.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {

/**
 * @brief Compute kernels for compute evolution of wavefield under the influence
 * of a source within a domain of elements specified by MediumTag.
 *
 * @tparam WavefieldType Type of wavefield. Forward or adjoint
 * @tparam DimensionType Dimension of the domain. 2D or 3D
 * @tparam MediumTag Tag specifying the medium
 * @tparam qp_type Type of quadrature points i.e. static or dynamic (will be
 * deprecated soon)
 */
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename qp_type>
class domain : public specfem::domain::impl::kernels::kernels<
                   WavefieldType, DimensionType, MediumTag, qp_type> {
public:
  constexpr static auto dimension = DimensionType; ///< Dimension of the domain
  constexpr static auto medium_tag = MediumTag;    ///< Medium tag
  constexpr static int num_dimensions =
      specfem::element::attributes<DimensionType, MediumTag>::dimension();
  constexpr static int components =
      specfem::element::attributes<DimensionType, MediumTag>::components();
  using quadrature_points_type = qp_type;

  /**
   * @brief Generate a new domain from an assembled mesh
   *
   * @param dt Time step
   * @param assembly Assembled mesh
   * @param quadrature_points Quadrature points object
   */
  domain(const type_real dt, const specfem::compute::assembly &assembly,
         const quadrature_points_type &quadrature_points)
      : field(assembly.fields.get_simulation_field<WavefieldType>()),
        specfem::domain::impl::kernels::kernels<
            WavefieldType, DimensionType, MediumTag, quadrature_points_type>(
            dt, assembly, quadrature_points) {}

  ~domain() = default;

  /**
   * @brief Invert the mass matrix
   */
  void invert_mass_matrix();

  /**
   * @brief Divide acceleration by mass matrix
   */
  void divide_mass_matrix();

private:
  specfem::compute::simulation_field<WavefieldType> field; ///< Wavefield
};
} // namespace domain

} // namespace specfem
