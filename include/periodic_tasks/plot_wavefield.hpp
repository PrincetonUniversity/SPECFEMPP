#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"
#include "enumerations/wavefield.hpp"
#include "plotter.hpp"
#include "specfem_mpi/interface.hpp"
#include <boost/filesystem.hpp>
#ifdef NO_VTK
#include <sstream>

#else
// Forward declarations for VTK classes
#include <vtkSmartPointer.h>
class vtkRenderer;
class vtkRenderWindow;
class vtkRenderWindowInteractor;
class vtkActor;
class vtkDataSetMapper;
class vtkUnstructuredGrid;
class vtkLookupTable;
class vtkNamedColors;
class vtkPoints;
class vtkCellArray;
class vtkFloatArray;
#endif

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Writer to plot the wavefield
 */
class plot_wavefield : public plotter {
public:
  /**
   * @brief Construct a new plotter object
   *
   * @param assembly SPECFFEM++ assembly object
   * @param output_format Output format of the plot (PNG, JPG, etc.)
   * @param component Component of the wavefield to plot (displacement,
   * velocity, etc.)
   * @param wavefield Type of wavefield to plot (forward, adjoint, etc.)
   * @param time_interval Time interval between subsequent plots
   * @param output_folder Path to output folder where plots will be stored
   */
  plot_wavefield(const specfem::compute::assembly &assembly,
                 const specfem::display::format &output_format,
                 const specfem::display::wavefield &component,
                 const specfem::wavefield::simulation_field &wavefield,
                 const int &time_interval,
                 const boost::filesystem::path &output_folder,
                 specfem::MPI::MPI *mpi);

  /**
   * @brief Updates the wavefield within open window
   *
   */
  void run(specfem::compute::assembly &assembly, const int istep) override;

  /**
   * @brief Wavefield plotter
   *
   * Opens a window, creates the grid and plots the materials.
   *
   * @param assembly SPECFFEM++ assembly object
   */
  void initialize(specfem::compute::assembly &assembly) override;

  /**
   * @brief Finalize the plotter
   *
   * Closes the window and cleans up resources.
   *
   * @param assembly SPECFFEM++ assembly object
   */
  void finalize(specfem::compute::assembly &assembly) override;

private:
  const specfem::display::format output_format; ///< Output format of the plot
  const specfem::display::wavefield component;  ///< Component of the wavefield
  const specfem::wavefield::simulation_field wavefield; ///< Type of wavefield
                                                        ///< to plot
  const boost::filesystem::path output_folder; ///< Path to output folder
  specfem::compute::assembly assembly;         ///< Assembly object

  // Grid parameter members
  int nspec;
  int ngllx;
  int ngllz;

  // MPI object
  specfem::MPI::MPI *mpi;

#ifndef NO_VTK

  // VTK objects that need to persist between calls
  vtkSmartPointer<vtkRenderer> renderer;
  vtkSmartPointer<vtkRenderWindow> render_window;
  vtkSmartPointer<vtkRenderWindowInteractor> render_window_interactor;
  vtkSmartPointer<vtkActor> material_actor;
  vtkSmartPointer<vtkActor> actor;
  vtkSmartPointer<vtkActor> outlineActor;
  vtkSmartPointer<vtkDataSetMapper> material_mapper;
  vtkSmartPointer<vtkDataSetMapper> wavefield_mapper;
  vtkSmartPointer<vtkUnstructuredGrid> unstructured_grid;
  vtkSmartPointer<vtkLookupTable> lut;
  vtkSmartPointer<vtkNamedColors> colors;

  // Separated grid and wavefield functions
  void create_quad_grid();
  void create_biquad_grid();
  vtkSmartPointer<vtkFloatArray>
  compute_wavefield_scalars(specfem::compute::assembly &assembly);
  vtkSmartPointer<vtkDataSetMapper> map_materials_with_color();
  vtkSmartPointer<vtkUnstructuredGrid> get_wavefield_on_vtk_biquad_grid();
  vtkSmartPointer<vtkUnstructuredGrid> get_wavefield_on_vtk_quad_grid();
  double sigmoid(double x);

  // Get wavefield component from display component
  specfem::wavefield::type get_wavefield_component();

#endif // NO_VTK
};
} // namespace periodic_tasks
} // namespace specfem
