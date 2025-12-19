#pragma once

#include "enumerations/display.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/assembly.hpp"
#include "specfem/periodic_tasks/plot_wavefield.hpp"
#include "specfem/periodic_tasks/plotter.hpp"
#include "specfem_mpi/interface.hpp"
#include <boost/filesystem.hpp>

#ifdef NO_VTK
#include <sstream>

#else
// Forward declarations for VTK classes
#include <vtkSmartPointer.h>
class vtkUnstructuredGrid;
class vtkFloatArray;

#ifndef NO_HDF5
#include <hdf5.h>
#endif // NO_HDF5

#endif // NO_VTK

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Writer to plot the wavefield for 3D simulations
 */
template <>
class plot_wavefield<specfem::dimension::type::dim3>
    : public plotter<specfem::dimension::type::dim3> {
public:
  constexpr static specfem::dimension::type dimension_tag =
      specfem::dimension::type::dim3;

  /**
   * @brief Construct a new plotter object
   *
   * @param assembly SPECFEM++ assembly object
   * @param output_format Output format of the plot (only vtkhdf supported)
   * @param wavefield_type Type of the wavefield to plot (displacement,
   * velocity, etc.)
   * @param wavefield Type of wavefield to plot (forward, adjoint, etc.)
   * @param time_interval Time interval between subsequent plots
   * @param output_folder Path to output folder where plots will be stored
   */
  plot_wavefield(
      const specfem::assembly::assembly<specfem::dimension::type::dim3>
          &assembly,
      const specfem::display::format &output_format,
      const specfem::wavefield::type &wavefield_type,
      const specfem::wavefield::simulation_field &wavefield,
      const type_real &dt, const int &time_interval,
      const boost::filesystem::path &output_folder, specfem::MPI::MPI *mpi);

  /**
   * @brief Updates the wavefield and writes to HDF5 file
   *
   */
  void run(specfem::assembly::assembly<dimension_tag> &assembly,
           const int istep) override;

  /**
   * @brief Initialize the wavefield plotter
   *
   * Creates the Lagrange hexahedral grid and initializes HDF5 file.
   *
   * @param assembly SPECFEM++ assembly object
   */
  void
  initialize(specfem::assembly::assembly<dimension_tag> &assembly) override;

  /**
   * @brief Finalize the plotter
   *
   * Closes HDF5 resources and cleans up.
   *
   * @param assembly SPECFEM++ assembly object
   */
  void finalize(specfem::assembly::assembly<dimension_tag> &assembly) override;

  const specfem::display::format output_format;  ///< Output format of the plot
  const specfem::wavefield::type wavefield_type; ///< Type of the wavefield
  const specfem::wavefield::simulation_field wavefield; ///< Type of wavefield
                                                        ///< to plot
  const boost::filesystem::path output_folder; ///< Path to output folder
  specfem::assembly::assembly<dimension_tag> assembly; ///< Assembly object

  // Grid parameter members
  int nspec; ///< Number of elements
  int ngllx; ///< Number of GLL points in x direction per element
  int nglly; ///< Number of GLL points in y direction per element
  int ngllz; ///< Number of GLL points in z direction per element

  // MPI object
  specfem::MPI::MPI *mpi;

  type_real dt; ///< Time step

private:
#ifndef NO_VTK

  vtkSmartPointer<vtkUnstructuredGrid> unstructured_grid;

#ifndef NO_HDF5
  // VTK HDF5 file handling members
  std::string hdf5_filename; // Store filename for reopening
  int current_timestep;
  int numPoints;          // Number of points in grid
  int numCells;           // Number of cells in grid
  int numConnectivityIds; // Number of connectivity IDs
#endif

  // Grid creation and wavefield computation
  void create_lagrange_hex_grid();
  vtkSmartPointer<vtkFloatArray> compute_wavefield_scalars(
      specfem::assembly::assembly<dimension_tag> &assembly);

  // Get wavefield type from display type
  specfem::wavefield::type get_wavefield_type();

  template <specfem::display::format format>
  void initialize(vtkSmartPointer<vtkFloatArray> &scalars);

  void initialize_display(vtkSmartPointer<vtkFloatArray> &scalars);

  template <specfem::display::format format>
  void run(vtkSmartPointer<vtkFloatArray> &scalars, const int istep);

  void run_render(vtkSmartPointer<vtkFloatArray> &scalars);

#endif // NO_VTK
};

} // namespace periodic_tasks
} // namespace specfem
