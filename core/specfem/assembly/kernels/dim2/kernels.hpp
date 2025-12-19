#pragma once

#include "enumerations/interface.hpp"
#include "medium/kernels_container.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/impl/value_containers.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief 2D spectral element misfit kernels container for seismic inversion
 *
 * This template specialization provides comprehensive storage and management
 * for all types of misfit kernels in 2D spectral element simulations.
 *
 * **Physical Context:**
 * Misfit kernels represent the sensitivity of seismic observations (waveform
 * data) to material parameters. They are computed through the interaction of
 * forward and adjoint wavefields and form the gradient for iterative inversion
 * algorithms such as full waveform inversion (FWI).
 *
 * @code
 * // Initialize kernels container for 2D mesh
 * specfem::assembly::kernels<specfem::dimension::type::dim2> kernels(
 *     nspec, ngllz, ngllx, element_types);
 *
 * // Access elastic kernels for a specific point
 * auto elastic_kernels = kernels.template get_container<
 *     specfem::element::medium_tag::elastic_psv,
 *     specfem::element::property_tag::isotropic>();
 *
 * // Accumulate kernel values during adjoint simulation
 * add_on_device(index, point_kernels, kernels);
 *
 * // Copy final kernels to host for output
 * kernels.copy_to_host();
 * @endcode
 *
 * @see specfem::assembly::kernels Base template class
 * @see specfem::medium::kernels_container Individual medium kernel containers
 * @see specfem::point::kernels Point-wise kernel accessors
 */
template <>
struct kernels<specfem::dimension::type::dim2>
    : public impl::value_containers<specfem::dimension::type::dim2,
                                    specfem::medium::kernels_container> {
public:
  /**
   * @brief Default constructor
   *
   */
  kernels() = default;

  /**
   * @brief Construct 2D kernels container from mesh and element information.
   *
   * Initializes the complete kernels storage system for a 2D spectral element
   * mesh. The constructor allocates kernel storage for all medium types and
   * material property models present in the mesh, setting up the mapping
   * between global element indices and local kernel storage indices.
   *
   * @param nspec Total number of spectral elements in the 2D mesh
   * @param ngllz Number of Gauss-Lobatto-Legendre quadrature points in
   * z-direction
   * @param ngllx Number of Gauss-Lobatto-Legendre quadrature points in
   * x-direction
   * @param element_types Element classification container specifying medium and
   * property types
   *
   */
  kernels(const int nspec, const int ngllz, const int ngllx,
          const specfem::assembly::element_types<dimension_tag> &element_types);

  /**
   * @brief Copy all kernel data from device to host memory.
   *
   */
  void copy_to_host() {
    impl::value_containers<dimension_tag,
                           specfem::medium::kernels_container>::copy_to_host();
  }

  /**
   * @brief Copy all kernel data from host to device memory.
   *
   */
  void copy_to_device() {
    impl::value_containers<
        dimension_tag, specfem::medium::kernels_container>::copy_to_device();
  }
};

} // namespace specfem::assembly
