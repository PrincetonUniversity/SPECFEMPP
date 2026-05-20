#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/periodic_tasks/plot_wavefield.hpp"
#include "specfem/periodic_tasks/plotter.hpp"
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

#ifndef NO_HDF5
#include <hdf5.h>
#endif // NO_HDF5

#endif // NO_VTK

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Writer to plot the wavefield
 */
template <>
class plot_wavefield<specfem::element::dimension_tag::dim2>
    : public plotter<specfem::element::dimension_tag::dim2> {
public:
  /**
   * @brief Construct a new plotter object
   *
   * @param assembly SPECFFEM++ assembly object
   * @param output_format Output format of the plot (PNG, JPG, etc.)
   * @param wavefield_type Type of the wavefield to plot (displacement,
   * velocity, etc.)
   * @param simulation_wavefield_type Type of wavefield to plot (forward,
   * adjoint, etc.)
   * @param time_interval Time interval between subsequent plots
   * @param output_folder Path to output folder where plots will be stored
   */
  plot_wavefield(
      const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
          &assembly,
      const specfem::enums::display_format &output_format,
      const specfem::enums::wavefield &wavefield_type,
      const specfem::simulation::field_type &simulation_wavefield_type,
      const specfem::enums::display_component &component,
      const int &time_interval, const boost::filesystem::path &output_folder,
      const specfem::enums::elastic_wave elastic_wave,
      const specfem::enums::electromagnetic_wave electromagnetic_wave);

  /**
   * @brief Updates the wavefield within open window
   *
   */
  void run(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
               &assembly,
           const int istep) override;

  /**
   * @brief Wavefield plotter
   *
   * Opens a window, creates the grid and plots the materials.
   *
   * @param assembly SPECFFEM++ assembly object
   */
  void
  initialize(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                 &assembly) override;

  /**
   * @brief Finalize the plotter
   *
   * Closes the window and cleans up resources.
   *
   * @param assembly SPECFFEM++ assembly object
   */
  void
  finalize(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
               &assembly) override;

  const specfem::enums::display_format output_format; ///< Output format of the
                                                      ///< plot
  const specfem::enums::display_component component;  ///< Component to plot
  const specfem::enums::wavefield wavefield_type;     ///< Type of the wavefield
  const specfem::simulation::field_type
      simulation_wavefield_type; ///< Type of wavefield
                                 ///< to plot
  bool nonnegative_field;        ///< Whether the field is non-negative (e.g.,
                                 ///< displacement with magnitude)
  const boost::filesystem::path output_folder; ///< Path to output folder
  const specfem::enums::elastic_wave elastic_wave;
  const specfem::enums::electromagnetic_wave electromagnetic_wave;
  specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
      assembly; ///< Assembly object

  // Grid parameter members
  int nspec; ///< Number of elements
  int ngllx; ///< Number of GLL points in x direction per element
  int ngllz; ///< Number of GLL points in z direction per element

private:
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

#ifndef NO_HDF5
  // VTK HDF5 file handling members
  std::string hdf5_filename; // Store filename for reopening
  int current_timestep;
  int numPoints;          // Number of points in grid
  int numCells;           // Number of cells in grid
  int numConnectivityIds; // Number of connectivity IDs
#endif

  // Separated grid and wavefield functions
  void create_quad_grid();
  void create_biquad_grid();
  void create_lagrange_quad_grid();
  vtkSmartPointer<vtkFloatArray> compute_wavefield_scalars(
      specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
          &assembly);
  vtkSmartPointer<vtkDataSetMapper> map_materials_with_color();

  /**
   * @brief Sigmoid function for transparency control
   *
   * @param x coordinated
   * @param scale controls slope of the center of the sigmoid
   * @param offset controls the offset from 0 where the sigmoid is centered.
   * @return double
   */
  double sigmoid(double x, double scale = 20, double offset = 0.75);

  // Get wavefield type from display type
  specfem::enums::wavefield get_wavefield_type();

  template <specfem::enums::display_format format>
  void initialize(vtkSmartPointer<vtkFloatArray> &scalars);

  void initialize_display(vtkSmartPointer<vtkFloatArray> &scalars);

  template <specfem::enums::display_format format>
  void run(vtkSmartPointer<vtkFloatArray> &scalars, const int istep);

  // Friend function for rendering
  void run_render(vtkSmartPointer<vtkFloatArray> &scalars);

  // Helper function to get scalar value at a given point
  static float get_scalar_value_at_point(
      const Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
          &wavefield_data,
      const specfem::enums::wavefield &wavefield_type,
      const specfem::enums::elastic_wave &elastic_wave,
      const specfem::enums::display_component &component, const int ispec,
      const int iz, const int ix);

#endif // NO_VTK
};
} // namespace periodic_tasks
} // namespace specfem
