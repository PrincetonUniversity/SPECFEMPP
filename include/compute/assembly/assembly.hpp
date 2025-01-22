#ifndef _COMPUTE_ASSEMBLY_HPP
#define _COMPUTE_ASSEMBLY_HPP

#include "IO/reader.hpp"
#include "compute/boundaries/boundaries.hpp"
#include "compute/boundary_values/boundary_values.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/coupled_interfaces/coupled_interfaces.hpp"
#include "compute/fields/fields.hpp"
#include "compute/kernels/kernels.hpp"
#include "compute/properties/interface.hpp"
#include "compute/receivers/receivers.hpp"
#include "compute/sources/sources.hpp"
#include "enumerations/display.hpp"
#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "receiver/interface.hpp"
#include "source/interface.hpp"

namespace specfem {
/**
 * @brief Compute namespace defines data structures used to store data related
 * to finite element assembly.
 *
 * The data is organized in a manner that makes it effiecient to access when
 * computing finite element compute kernels.
 *
 */
namespace compute {
/**
 * @brief Finite element assembly data
 *
 */
struct assembly {
  specfem::compute::mesh mesh; ///< Properties of the assembled mesh
  specfem::compute::element_types element_types; ///< Element tags for every
                                                 ///< spectral element
  specfem::compute::partial_derivatives partial_derivatives; ///< Partial
                                                             ///< derivatives of
                                                             ///< the basis
                                                             ///< functions
  specfem::compute::properties properties; ///< Material properties
  specfem::compute::kernels kernels; ///< Frechet derivatives (Misfit kernels)
  specfem::compute::sources sources; ///< Source information
  specfem::compute::receivers receivers;   ///< Receiver information
  specfem::compute::boundaries boundaries; ///< Boundary conditions
  specfem::compute::coupled_interfaces coupled_interfaces; ///< Coupled
                                                           ///< interfaces
                                                           ///< between 2
                                                           ///< mediums
  specfem::compute::fields fields; ///< Displacement, velocity, and acceleration
                                   ///< fields
  specfem::compute::boundary_values boundary_values; ///< Field values at the
                                                     ///< boundaries

  /**
   * @brief Generate a finite element assembly
   *
   * @param mesh Finite element mesh as read from mesher
   * @param quadratures Quadrature points and weights
   * @param sources Source information
   * @param receivers Receiver information
   * @param stypes Types of seismograms
   * @param t0 Start time of simulation
   * @param dt Time step
   * @param max_timesteps Maximum number of time steps
   * @param max_sig_step Maximum number of seismogram time steps
   * @param nstep_between_samples Number of time steps between output seismogram
   * samples
   * @param simulation Type of simulation (forward, adjoint, etc.)
   * @param property_reader Reader for GLL model (skip material property
   * assignment if exists)
   */
  assembly(
      const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
      const specfem::quadrature::quadratures &quadratures,
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const std::vector<std::shared_ptr<specfem::receivers::receiver> >
          &receivers,
      const std::vector<specfem::enums::seismogram::type> &stypes,
      const type_real t0, const type_real dt, const int max_timesteps,
      const int max_sig_step, const int nsteps_between_samples,
      const specfem::simulation::type simulation,
      const std::shared_ptr<specfem::IO::reader> &property_reader);

  /**
   * @brief Maps the component of wavefield on the entire spectral element grid
   *
   * This field can be used to generate a plot of the wavefield
   *
   * @param component Component of the wavefield to map
   * @return Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Wavefield mapped on the entire grid. Dimensions of the view are nspec,
   * ngllz, ngllx
   */
  Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
  generate_wavefield_on_entire_grid(
      const specfem::wavefield::simulation_field wavefield,
      const specfem::wavefield::type component);
};

} // namespace compute
} // namespace specfem

#endif
