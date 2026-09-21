#pragma once

#include "specfem/assembly/attenuation.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/boundary_values.hpp"
#include "specfem/assembly/conforming_interfaces.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/field_derivative_storage.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/info.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/kernels.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/assembly/receivers.hpp"
#include "specfem/assembly/sources.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/receivers.hpp"
#include "specfem/source.hpp"

namespace specfem::io {
class reader;
}

namespace specfem::assembly {

/**
 * @brief Specialization of the assembly class for 3D SEM simulations
 *
 * Provides 3D specializations for containers used to store simulation data
 * required for & computed during 3D SEM simulations
 *
 */
template <> struct assembly<specfem::element::dimension_tag::dim3> {

  /**
   * @name Public Constants
   *
   */
  ///@{
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag
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

  specfem::assembly::element_intersections<dimension_tag>
      element_intersections; ///< Element intersections for different types of
                             ///< interfaces between two media (e.g.,
                             ///< fluid-solid interface).

  /**
   * @brief Partial derivatives of the basis functions at every quadrature point
   *
   */
  specfem::assembly::jacobian_matrix<dimension_tag> jacobian_matrix;

  /**
   * @brief Storage for field derivatives at every quadrature point for elements
   * that require it (e.g., attenuating elements).
   */
  specfem::assembly::FieldDerivativeStorage<dimension_tag>
      field_derivative_storage;

  /**
   * @brief Attenuation properties for the mesh at every quadrature point
   */
  specfem::assembly::Attenuation<dimension_tag> attenuation;

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
   * @brief Information about receivers, locations, seismogram types,
   * lagrange interpolation, etc.
   *
   */
  specfem::assembly::receivers<dimension_tag> receivers;

  /**
   * @brief Wavefield values at every distinct quadrature point in the mesh,
   * \f$(s, \partial s / \partial t, \partial^2 s /\partial t^2)\f$
   */
  specfem::assembly::fields<dimension_tag> fields;

  specfem::assembly::boundaries<dimension_tag> boundaries; ///< Boundary
                                                           ///< conditions
  specfem::assembly::conforming_interfaces<dimension_tag>
      conforming_interfaces; ///< Conforming interfaces between different media
                             ///< (e.g., fluid-solid interface)

  specfem::assembly::boundary_values<dimension_tag>
      boundary_values; ///< Field
                       ///< values at
                       ///< the
                       ///< boundaries

  specfem::assembly::Info<dimension_tag> info; ///< Information about the mesh
                                               ///< and simulation

  specfem::assembly::mpi<dimension_tag> mpi_interfaces; ///< MPI communication
                                                        ///< groups for face
                                                        ///< data exchange
                                                        ///< between partitions

  type_real t0; ///< Simulation start time

  type_real dt; ///< Time step

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
   * @param flux_scheme_config Flux scheme rules for nonconforming interfaces
   */
  template <specfem::simulation::model ModelTag>
  assembly(const specfem::mesh::mesh<ModelTag> &mesh,
           const specfem::quadrature::quadratures &quadratures,
           std::vector<std::shared_ptr<specfem::sources::source<dimension_tag>>>
               &sources,
           const std::vector<
               std::shared_ptr<specfem::receivers::receiver<dimension_tag>>>
               &receivers,
           const std::vector<specfem::enums::wavefield> &stypes,
           const type_real t0, const type_real dt, const int max_timesteps,
           const int max_sig_step, const int nsteps_between_samples,
           const specfem::simulation::type simulation,
           const bool allocate_boundary_values,
           const std::shared_ptr<specfem::io::reader> &property_reader,
           const specfem::element_coupling::flux_scheme_configuration
               &flux_scheme_config =
                   specfem::element_coupling::flux_scheme_configuration());

  assembly() = default;

  /**
   * @brief Maps the component of wavefield on the entire spectral element grid
   *
   * This field can be used to generate a plot of the wavefield
   *
   * @param component Component of the wavefield to map
   * @return Kokkos::View<type_real *****, Kokkos::LayoutLeft,
   * Kokkos::HostSpace> Wavefield mapped on the entire grid. Dimensions of the
   * view are nspec, ngllz, nglly, ngllx, ncomponents
   */
  Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>
  generate_wavefield_on_entire_grid(
      const specfem::simulation::field_type wavefield,
      const specfem::enums::wavefield component);

  /**
   * @brief Get the total number of spectral elements in the mesh
   * @return int Total number of spectral elements
   */
  int get_total_number_of_elements() const { return mesh.nspec; }

  /**
   * @brief Get the total number of degrees of freedom in the mesh
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
