#pragma once

#include "enumerations/interface.hpp"
#include "io/reader.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/boundary_values.hpp"
#include "specfem/assembly/compute_source_array.hpp"
#include "specfem/assembly/conforming_interfaces.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/kernels.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/nonconforming_interfaces.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/assembly/receivers.hpp"
#include "specfem/assembly/sources.hpp"
#include "specfem/mesh.hpp"
#include "specfem/receivers.hpp"
#include "specfem/source.hpp"

namespace specfem::assembly {
/**
 * @brief Specialization of the assembly class for 2D SEM simulations
 *
 * Provides 2D specializations for containers used to store simulation data
 * required for & computed during 2D SEM simulations
 *
 */
template <> struct assembly<specfem::dimension::type::dim2> {

  /**
   * @name Public Constants
   *
   */
  ///@{
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag
  ///@}

  /** @name Data Containers
   *
   * Data containers used to store computation data required for different terms
   * in the constitutive equations
   */
  ///@{

  /**
   * @brief Properties of the assembled mesh
   *
   */
  specfem::assembly::mesh<dimension_tag> mesh;

  /**
   * @brief Element types for every spectral element in the mesh
   *
   */
  specfem::assembly::element_types<dimension_tag> element_types;

  /**
   * @brief Edge types for every edge on coupled interface in the mesh.
   *
   * The edge type defines the flux scheme to used when computing coupling terms
   * between two media (e.g., fluid-solid interface).
   *
   */
  specfem::assembly::edge_types<dimension_tag> edge_types;

  /**
   * @brief Partial derivatives of the basis functions at every quadrature point
   *
   */
  specfem::assembly::jacobian_matrix<dimension_tag> jacobian_matrix;

  /**
   * @brief Material properties for the mesh at every quadrature point
   *
   */
  specfem::assembly::properties<dimension_tag> properties;

  /**
   * @brief Misfit kernels (Frechet derivatives) computed at every quadrature
   * point during adjoint simulations. The container is empty for forward
   * simulations.
   *
   */
  specfem::assembly::kernels<dimension_tag> kernels;

  /**
   * @brief Information about sources, locations, source time functions,
   * lagrange interpolation, etc.
   *
   */
  specfem::assembly::sources<dimension_tag> sources;

  /**
   * @brief Information about receivers, locations, seismogram types, lagrange
   * interpolation, etc.
   *
   */
  specfem::assembly::receivers<dimension_tag> receivers;

  /**
   * @brief Information about boundary conditions in the mesh.
   *
   * The container stores data required to implement different types of boundary
   * conditions (e.g., for stacey boudary conditions, we store normal vectors &
   * weight factors at every quadrature point on the boundary).
   *
   */
  specfem::assembly::boundaries<dimension_tag> boundaries;

  /**
   * @brief Information about conforming interfaces between 2 media in the mesh.
   *
   * The container stores data required to implement coupling terms between 2
   * media (e.g., fluid-solid interface).
   *
   */
  specfem::assembly::conforming_interfaces<dimension_tag> conforming_interfaces;

  /**
   * @brief Information about non-conforming interfaces between 2 media in the
   * mesh.
   *
   * The container stores data required to implement coupling terms between 2
   * media (e.g., fluid-solid interface).
   *
   */
  specfem::assembly::nonconforming_interfaces<dimension_tag>
      nonconforming_interfaces;

  /**
   * @brief Wavefield values at every distinct quadrature point in the mesh,
   * \f$(s, \partial s / \partial t, \partial^2 s /\partial t^2)\f$
   */
  specfem::assembly::fields<dimension_tag> fields;

  /**
   * @brief Field values at the boundaries at every time step.
   *
   * This container stores the wavefield values at the (stacey) boundaries
   * computed during forward simulations. The values are then used during
   * adjoint simulations to impose boundary condition on the adjoint wavefield.
   * The container is empty if wavefield writer is disabled.
   *
   */
  specfem::assembly::boundary_values<dimension_tag>
      boundary_values; ///< Field
                       ///< values at
                       ///< the
                       ///< boundaries

  ///@}

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
   * @param write_wavefield Whether to write wavefield
   * @param property_reader Reader for GLL model (skip material property
   * assignment if exists)
   */
  assembly(
      const specfem::mesh::mesh<dimension_tag> &mesh,
      const specfem::quadrature::quadratures &quadratures,
      std::vector<std::shared_ptr<specfem::sources::source<dimension_tag> > >
          &sources,
      const std::vector<
          std::shared_ptr<specfem::receivers::receiver<dimension_tag> > >
          &receivers,
      const std::vector<specfem::wavefield::type> &stypes, const type_real t0,
      const type_real dt, const int max_timesteps, const int max_sig_step,
      const int nsteps_between_samples,
      const specfem::simulation::type simulation,
      const bool allocate_boundary_values,
      const std::shared_ptr<specfem::io::reader> &property_reader);

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

  /**
   * @brief Get the total number of spectral elements in the mesh
   *
   * @return int Total number of spectral elements
   */
  int get_total_number_of_elements() const { return mesh.nspec; }

  /**
   * @brief Get the total number of degrees of freedom in the mesh
   *
   * @return int Total number of degrees of freedom
   */
  int get_total_degrees_of_freedom() {
    return fields.buffer.get_total_degrees_of_freedom();
  }

  /**
   * @brief Print assembly information
   *
   * Generates a formatted string containing relevant information about the
   * assembly. This information is logged into the output of the simulation.
   *
   * @return std::string Assembly information as a string
   */
  std::string print() const;

  /**
   * @brief Check if Jacobian for any spectral element in the mesh is smaller
   * than some threshold
   *
   * This function throws a runtime error if the Jacobian is smaller than 1e-10
   * If VTK is enabled, it also generates a plot of the spectral elements with
   * small Jacobian
   *
   */
  void check_jacobian_matrix() const;
};

} // namespace specfem::assembly
