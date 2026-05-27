#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/periodic_tasks/plot_wavefield.hpp"
#include "specfem/periodic_tasks/plotter.hpp"
#include <boost/filesystem.hpp>
#include <vector>

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
class plot_wavefield<specfem::element::dimension_tag::dim3>
    : public plotter<specfem::element::dimension_tag::dim3> {
public:
  constexpr static specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim3;

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
      const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
          &assembly,
      const specfem::enums::display_format &output_format,
      const specfem::enums::wavefield &wavefield_type,
      const specfem::simulation::field_type &simulation_wavefield_type,
      const specfem::enums::display_component &component,
      const int &time_interval, const boost::filesystem::path &output_folder);

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

  const specfem::enums::display_format output_format; ///< Output format of the
                                                      ///< plot
  const specfem::enums::wavefield wavefield_type;     ///< Type of the wavefield
  const specfem::simulation::field_type
      simulation_wavefield_type;                     ///< Type of wavefield
                                                     ///< to plot
  const specfem::enums::display_component component; ///< Component of the
                                                     ///< wavefield to plot
  const boost::filesystem::path output_folder;       ///< Path to output folder
  const specfem::assembly::assembly<dimension_tag> &assembly; ///< Assembly
                                                              ///< object

  // Grid parameter members
  int nspec; ///< Number of elements
  int ngllx; ///< Number of GLL points in x direction per element
  int nglly; ///< Number of GLL points in y direction per element
  int ngllz; ///< Number of GLL points in z direction per element

private:
#ifndef NO_VTK

  vtkSmartPointer<vtkUnstructuredGrid> unstructured_grid;

#ifndef NO_HDF5
  // VTK HDF5 file handling members
  std::string hdf5_filename;    ///< Store filename for reopening
  int current_timestep;         ///< Current output timestep index
  long long numPoints;          ///< Number of local points in grid
  long long numCells;           ///< Number of local cells in grid
  long long numConnectivityIds; ///< Number of local connectivity IDs

  // MPI partition info (all zero/local in serial builds)
  long long global_point_offset = 0;        ///< This rank's offset into global
                                            ///< Points array
  long long global_cell_offset = 0;         ///< This rank's offset into global
                                            ///< Cells array
  long long global_connectivity_offset = 0; ///< This rank's offset into global
                                            ///< Connectivity array
  long long total_points = 0;       ///< Sum of numPoints across all ranks
  long long total_cells = 0;        ///< Sum of numCells across all ranks
  long long total_connectivity = 0; ///< Sum of numConnectivityIds across all
                                    ///< ranks
  int num_parts = 1;                ///< Number of MPI ranks (1 for serial)
  bool use_parallel_hdf5 = false;   ///< True if using collective parallel I/O
  std::vector<long long> all_point_offsets; ///< Per-rank point offsets (rank 0)
  std::vector<long long> all_point_counts;  ///< Per-rank point counts (rank 0)
  std::vector<long long> all_cell_counts;   ///< Per-rank cell counts (rank 0)
  std::vector<long long> all_connectivity_counts; ///< Per-rank connectivity
                                                  ///< counts (rank 0)
#endif

  // Grid creation and wavefield computation
  void create_lagrange_hex_grid();
  vtkSmartPointer<vtkFloatArray> compute_wavefield_scalars(
      specfem::assembly::assembly<dimension_tag> &assembly);

  // Get wavefield type from display type
  specfem::enums::wavefield get_wavefield_type();

  template <specfem::enums::display_format format>
  void initialize(vtkSmartPointer<vtkFloatArray> &scalars);

  void initialize_display(vtkSmartPointer<vtkFloatArray> &scalars);

  template <specfem::enums::display_format format>
  void run(vtkSmartPointer<vtkFloatArray> &scalars, const int istep);

  void run_render(vtkSmartPointer<vtkFloatArray> &scalars);

  // Helper to extend a 1D HDF5 dataset and write a single scalar value.
  // When do_write is false, extends the dataset but writes nothing (for
  // collective H5Dset_extent on non-writing ranks in parallel HDF5).
  static void extend_and_write_scalar(hid_t parent, const char *dataset_name,
                                      hsize_t new_extent, hsize_t write_offset,
                                      hid_t mem_type, const void *data,
                                      hid_t dxpl = H5P_DEFAULT,
                                      bool do_write = true);

  // Helper to extend a 1D HDF5 dataset and write an array of values.
  // When do_write is false, extends the dataset but writes nothing.
  static void extend_and_write_array(hid_t parent, const char *dataset_name,
                                     hsize_t new_extent, hsize_t write_offset,
                                     hsize_t count, hid_t mem_type,
                                     const void *data, hid_t dxpl = H5P_DEFAULT,
                                     bool do_write = true);

  /// @brief Compute MPI offsets via prefix sum (no-op for serial)
  void compute_mpi_offsets();

  /// @brief Create HDF5 file access property list (parallel or serial)
  hid_t create_file_access_plist() const;

  // Helper function to get scalar value at a given point
  static float get_scalar_value_at_point(
      const Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>
          &wavefield_data,
      const specfem::enums::wavefield &wavefield_type,
      const specfem::enums::display_component &component, const int ispec,
      const int iz, const int iy, const int ix);

#endif // NO_VTK
};

} // namespace periodic_tasks
} // namespace specfem
