#pragma once

#include "enumerations/interface.hpp"
#include "io/reader.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/boundary_values.hpp"
#include "specfem/assembly/compute_source_array.hpp"
#include "specfem/assembly/coupled_interfaces.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/kernels.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/assembly/receivers.hpp"
#include "specfem/assembly/sources.hpp"
#include "specfem/receivers.hpp"
#include "specfem/source.hpp"

/**
 * @brief Assembly namespace defines data structures used to store data related
 * to finite element assembly.
 *
 * The data is organized in a manner that makes it effiecient to access when
 * computing finite element compute kernels.
 *
 */
namespace specfem::assembly {
/**
 * @brief Finite element assembly data
 *
 */
template <> struct assembly<specfem::dimension::type::dim2> {

  constexpr static auto dimension_tag = specfem::dimension::type::dim2;

  specfem::assembly::mesh<dimension_tag> mesh; ///< Properties of the assembled
                                               ///< mesh
  specfem::assembly::element_types<dimension_tag> element_types; ///< Element
                                                                 ///< tags for
                                                                 ///< every
                                                                 ///< spectral
                                                                 ///< element

  specfem::assembly::edge_types<dimension_tag> edge_types;
  specfem::assembly::jacobian_matrix<dimension_tag>
      jacobian_matrix;                                     ///< Partial
                                                           ///< derivatives
                                                           ///< of the
                                                           ///< basis
                                                           ///< functions
  specfem::assembly::properties<dimension_tag> properties; ///< Material
                                                           ///< properties
  specfem::assembly::kernels<dimension_tag> kernels; ///< Frechet derivatives
                                                     ///< (Misfit kernels)
  specfem::assembly::sources<dimension_tag> sources; ///< Source information
  specfem::assembly::receivers<dimension_tag> receivers;   ///< Receiver
                                                           ///< information
  specfem::assembly::boundaries<dimension_tag> boundaries; ///< Boundary
                                                           ///< conditions
  specfem::assembly::coupled_interfaces<dimension_tag>
      coupled_interfaces; ///< Coupled interfaces between 2 mediums (new
                          ///< implementation)
  specfem::assembly::fields<dimension_tag> fields; ///< Displacement, velocity,
                                                   ///< and acceleration fields
  specfem::assembly::boundary_values<dimension_tag>
      boundary_values; ///< Field
                       ///< values at
                       ///< the
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

  int get_total_number_of_elements() const { return mesh.nspec; }

  int get_total_degrees_of_freedom() {
    return fields.buffer.get_total_degrees_of_freedom();
  }

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
