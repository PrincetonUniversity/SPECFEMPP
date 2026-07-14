#include "plot_wavefield.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi/mpi.hpp"
#include "specfem/periodic_tasks/plotter.hpp"
#include "specfem/program.hpp"
#include "specfem/utilities.hpp"

#ifdef NO_VTK

#include <sstream>

#else

#include <boost/filesystem.hpp>
#include <cmath>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkLagrangeHexahedron.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

#ifndef NO_HDF5
#include "specfem/io_backends/HDF5/impl/h5_check.hpp"

namespace specfem::periodic_tasks::plot_wavefield_impl {

/// @brief Create a 1D dataset and write data, handling parallel/serial I/O
inline void write_static_1d(hid_t parent, const char *name, hid_t mem_type,
                            hsize_t total_size, hsize_t local_offset,
                            hsize_t local_count, const void *data,
                            bool use_parallel, hid_t dxpl) {
  hsize_t dims[1] = { total_size };
  hid_t dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
  hid_t dataset =
      SPECFEM_H5_CHECK_ID(H5Dcreate(parent, name, mem_type, dataspace,
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  if (use_parallel) {
    SPECFEM_H5_CHECK(H5Sselect_hyperslab(
        dataspace, H5S_SELECT_SET, &local_offset, NULL, &local_count, NULL));
    hid_t memspace =
        SPECFEM_H5_CHECK_ID(H5Screate_simple(1, &local_count, NULL));
    SPECFEM_H5_CHECK(
        H5Dwrite(dataset, mem_type, memspace, dataspace, dxpl, data));
    SPECFEM_H5_CHECK(H5Sclose(memspace));
  } else {
    SPECFEM_H5_CHECK(
        H5Dwrite(dataset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
  }
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(dataspace));
}

/// @brief Create a 2D dataset and write data, handling parallel/serial I/O
inline void write_static_2d(hid_t parent, const char *name, hid_t mem_type,
                            hsize_t total_rows, hsize_t ncols,
                            hsize_t row_offset, hsize_t row_count,
                            const void *data, bool use_parallel, hid_t dxpl) {
  hsize_t dims[2] = { total_rows, ncols };
  hid_t dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(2, dims, NULL));
  hid_t dataset =
      SPECFEM_H5_CHECK_ID(H5Dcreate(parent, name, mem_type, dataspace,
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  if (use_parallel) {
    hsize_t offset[2] = { row_offset, 0 };
    hsize_t count[2] = { row_count, ncols };
    SPECFEM_H5_CHECK(H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset,
                                         NULL, count, NULL));
    hid_t memspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(2, count, NULL));
    SPECFEM_H5_CHECK(
        H5Dwrite(dataset, mem_type, memspace, dataspace, dxpl, data));
    SPECFEM_H5_CHECK(H5Sclose(memspace));
  } else {
    SPECFEM_H5_CHECK(
        H5Dwrite(dataset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
  }
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(dataspace));
}

} // namespace specfem::periodic_tasks::plot_wavefield_impl

#endif // NO_HDF5

#endif // NO_VTK

#ifdef NO_VTK

// Add constructor implementation for NO_VTK builds
specfem::periodic_tasks::plot_wavefield<specfem::element::dimension_tag::dim3>::
    plot_wavefield(
        const specfem::assembly::assembly<dimension_tag> &assembly,
        const specfem::enums::display_format &output_format,
        const specfem::enums::wavefield &wavefield_type,
        const specfem::simulation::field_type &simulation_wavefield_type,
        const specfem::enums::display_component &component,
        const int &time_interval, const boost::filesystem::path &output_folder)
    : assembly(assembly), simulation_wavefield_type(simulation_wavefield_type),
      wavefield_type(wavefield_type), component(component),
      plotter<dimension_tag>(time_interval), output_format(output_format),
      output_folder(output_folder), nspec(assembly.mesh.nspec),
      ngllx(assembly.mesh.element_grid.ngllx),
      nglly(assembly.mesh.element_grid.nglly),
      ngllz(assembly.mesh.element_grid.ngllz) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::run(
        specfem::assembly::assembly<dimension_tag> &assembly, const int istep) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::initialize(
        specfem::assembly::assembly<dimension_tag> &assembly) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::finalize(
        specfem::assembly::assembly<dimension_tag> &assembly) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

#else

// Constructor
specfem::periodic_tasks::plot_wavefield<specfem::element::dimension_tag::dim3>::
    plot_wavefield(
        const specfem::assembly::assembly<dimension_tag> &assembly,
        const specfem::enums::display_format &output_format,
        const specfem::enums::wavefield &wavefield_type,
        const specfem::simulation::field_type &simulation_wavefield_type,
        const specfem::enums::display_component &component,
        const int &time_interval, const boost::filesystem::path &output_folder)
    : assembly(assembly), simulation_wavefield_type(simulation_wavefield_type),
      wavefield_type(wavefield_type), component(component),
      plotter<dimension_tag>(time_interval), output_format(output_format),
      output_folder(output_folder), nspec(assembly.mesh.nspec),
      ngllx(assembly.mesh.element_grid.ngllx),
      nglly(assembly.mesh.element_grid.nglly),
      ngllz(assembly.mesh.element_grid.ngllz) {
  // Only VTK HDF5 output is supported for 3D
  if (output_format != specfem::enums::display_format::vtkhdf) {
    throw std::runtime_error(
        "Only VTK HDF5 output format is supported for 3D wavefield plotting");
  }
};

// Helper function to get scalar value at a given point
float specfem::periodic_tasks::plot_wavefield<
    specfem::element::dimension_tag::dim3>::
    get_scalar_value_at_point(
        const Kokkos::View<type_real *****, Kokkos::LayoutLeft,
                           Kokkos::HostSpace> &wavefield_data,
        const specfem::enums::wavefield &wavefield_type,
        const specfem::enums::display_component &component, const int ispec,
        const int iz, const int iy, const int ix) {

  if (wavefield_type == specfem::enums::wavefield::pressure ||
      wavefield_type == specfem::enums::wavefield::rotation ||
      wavefield_type == specfem::enums::wavefield::intrinsic_rotation ||
      wavefield_type == specfem::enums::wavefield::curl) {
    return std::abs(wavefield_data(ispec, iz, iy, ix, 0));
  }

  // Computing the component or magnitude for vector fields
  if (component == specfem::enums::display_component::x) {
    return wavefield_data(ispec, iz, iy, ix, 0);
  } else if (component == specfem::enums::display_component::y) {
    return wavefield_data(ispec, iz, iy, ix, 1);
  } else if (component == specfem::enums::display_component::z) {
    return wavefield_data(ispec, iz, iy, ix, 2);
  } else if (component == specfem::enums::display_component::magnitude) {
    // Compute magnitude from 3-component vector
    type_real magnitude = 0.0;
    for (int icomp = 0; icomp < 3; ++icomp) {
      const type_real comp = wavefield_data(ispec, iz, iy, ix, icomp);
      magnitude += comp * comp;
    }
    return static_cast<float>(std::sqrt(magnitude));
  } else {
    throw std::runtime_error("Invalid component,'" +
                             specfem::enums::to_string(component) +
                             "', for wavefield plotting in 3D.");
  }
}

/**
 * @brief Create a Lagrange hexahedral grid using all GLL points
 *
 * This function creates a higher-order Lagrange hexahedral element for each
 * spectral element, using all GLL points as control points. Each spectral
 * element becomes one vtkLagrangeHexahedron with (ngllx * nglly * ngllz)
 * control points.
 *
 * For ngll = 5, each spectral element becomes one vtkLagrangeHexahedron
 * with 125 control points arranged in a structured 5x5x5 grid.
 */
void specfem::periodic_tasks::plot_wavefield<
    specfem::element::dimension_tag::dim3>::create_lagrange_hex_grid() {
  const auto &coordinates = assembly.mesh.h_coord;

  // Each spectral element becomes one Lagrange hexahedron
  const int points_per_element = ngllx * nglly * ngllz;

  auto points = vtkSmartPointer<vtkPoints>::New();
  auto cells = vtkSmartPointer<vtkCellArray>::New();

  int point_counter = 0;

  // Loop over spectral elements
  for (int ispec = 0; ispec < nspec; ++ispec) {
    auto lagrange_hex = vtkSmartPointer<vtkLagrangeHexahedron>::New();

    // Set the order of the Lagrange hexahedron
    // Order is ngll-1 in each direction
    lagrange_hex->SetOrder(ngllx - 1, nglly - 1, ngllz - 1);
    lagrange_hex->GetPointIds()->SetNumberOfIds(points_per_element);

    // Add all GLL points for this element
    // VTK Lagrange hexahedron uses PointIndexFromIJK(i, j, k) for ordering
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int iy = 0; iy < nglly; ++iy) {
        for (int ix = 0; ix < ngllx; ++ix) {
          // Get the local index in VTK's Lagrange ordering
          int local_idx = lagrange_hex->PointIndexFromIJK(ix, iy, iz);

          // Add the point coordinates
          // Note: 3D coords are stored as (ispec, iz, iy, ix, idim)
          points->InsertNextPoint(coordinates(ispec, iz, iy, ix, 0),
                                  coordinates(ispec, iz, iy, ix, 1),
                                  coordinates(ispec, iz, iy, ix, 2));

          // Set the point ID in the cell
          lagrange_hex->GetPointIds()->SetId(local_idx, point_counter);
          point_counter++;
        }
      }
    }

    // Add the cell
    cells->InsertNextCell(lagrange_hex);
  }

  // Create the unstructured grid
  unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  unstructured_grid->SetPoints(points);
  unstructured_grid->SetCells(VTK_LAGRANGE_HEXAHEDRON, cells);
}

// Compute wavefield scalar values for the grid points
vtkSmartPointer<vtkFloatArray>
specfem::periodic_tasks::plot_wavefield<specfem::element::dimension_tag::dim3>::
    compute_wavefield_scalars(
        specfem::assembly::assembly<dimension_tag> &assembly) {
  const auto wavefield_type = this->wavefield_type;
  const auto &wavefield_data = assembly.generate_wavefield_on_entire_grid(
      this->simulation_wavefield_type, wavefield_type);

  auto scalars = vtkSmartPointer<vtkFloatArray>::New();

  // For Lagrange hexahedral grid
  if (unstructured_grid->GetCellType(0) == VTK_LAGRANGE_HEXAHEDRON) {
    // Loop over spectral elements
    for (int ispec = 0; ispec < nspec; ++ispec) {
      // For each point in the element, compute scalar using helper function
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix) {
            scalars->InsertNextValue(
                get_scalar_value_at_point(wavefield_data, wavefield_type,
                                          this->component, ispec, iz, iy, ix));
          }
        }
      }
    }
  } else {
    throw std::runtime_error(
        "Unsupported grid type for wavefield scalar computation in 3D.");
  }

  return scalars;
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::initialize_vtkhdf(
        vtkSmartPointer<vtkFloatArray> &scalars) {

#ifndef NO_HDF5

  // Initialize VTK HDF5 file for time series output
  this->current_timestep = 0;
  this->numPoints = 0;
  this->numCells = 0;

  // Determine filename and parallel mode
#ifdef SPECFEM_HDF5_IS_PARALLEL
  this->use_parallel_hdf5 = true;
  this->hdf5_filename = (this->output_folder / "wavefield.vtkhdf").string();
#elif defined(SPECFEM_ENABLE_MPI)
  this->use_parallel_hdf5 = false;
  this->hdf5_filename = specfem::MPI::format_proc_filename(
      (this->output_folder / "wavefield.vtkhdf").string());
  // Create the proc subdirectory on rank 0, then synchronize
  if (specfem::MPI::main_proc()) {
    boost::filesystem::create_directories(
        boost::filesystem::path(this->hdf5_filename).parent_path());
  }
  specfem::MPI::sync();
#else
  this->use_parallel_hdf5 = false;
  this->hdf5_filename = (this->output_folder / "wavefield.vtkhdf").string();
#endif

  // Create HDF5 file with appropriate access property list
  hid_t fapl = this->create_file_access_plist();
  hid_t hdf5_file_id = SPECFEM_H5_CHECK_ID(
      H5Fcreate(this->hdf5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl));
  SPECFEM_H5_CHECK(H5Pclose(fapl));
  hid_t vtkhdf_group = SPECFEM_H5_CHECK_ID(H5Gcreate(
      hdf5_file_id, "/VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  // Set VTKHDF attributes
  {
    int version[2] = { 2, 0 };
    hsize_t dims[1] = { 2 };
    hid_t attr_space = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
    hid_t attr =
        SPECFEM_H5_CHECK_ID(H5Acreate(vtkhdf_group, "Version", H5T_NATIVE_INT,
                                      attr_space, H5P_DEFAULT, H5P_DEFAULT));
    SPECFEM_H5_CHECK(H5Awrite(attr, H5T_NATIVE_INT, version));
    SPECFEM_H5_CHECK(H5Aclose(attr));
    SPECFEM_H5_CHECK(H5Sclose(attr_space));

    // Set Type attribute
    hid_t str_type = SPECFEM_H5_CHECK_ID(H5Tcopy(H5T_C_S1));
    SPECFEM_H5_CHECK(H5Tset_size(str_type, 16));
    attr_space = SPECFEM_H5_CHECK_ID(H5Screate(H5S_SCALAR));
    attr = SPECFEM_H5_CHECK_ID(H5Acreate(vtkhdf_group, "Type", str_type,
                                         attr_space, H5P_DEFAULT, H5P_DEFAULT));
    const char *type_str = "UnstructuredGrid";
    SPECFEM_H5_CHECK(H5Awrite(attr, str_type, type_str));
    SPECFEM_H5_CHECK(H5Aclose(attr));
    SPECFEM_H5_CHECK(H5Sclose(attr_space));
    SPECFEM_H5_CHECK(H5Tclose(str_type));
  }

  // Write static geometry to HDF5 file
  this->numPoints = this->unstructured_grid->GetNumberOfPoints();
  this->numCells = this->unstructured_grid->GetNumberOfCells();

  // Extract connectivity
  vtkCellArray *cells_vtkh = this->unstructured_grid->GetCells();
  vtkIdType npts;
  const vtkIdType *pts;
  std::vector<long long> connectivity;
  std::vector<long long> offsets;
  std::vector<unsigned char> types;

  offsets.push_back(0);
  for (vtkIdType i = 0; i < this->numCells; i++) {
    cells_vtkh->GetCellAtId(i, npts, pts);
    for (vtkIdType j = 0; j < npts; j++) {
      connectivity.push_back(pts[j]);
    }
    offsets.push_back(connectivity.size());
    types.push_back(this->unstructured_grid->GetCellType(i));
  }

  // Store connectivity size for later use
  this->numConnectivityIds = connectivity.size();

  // Compute MPI offsets (no-op for serial builds)
  this->compute_mpi_offsets();

  // For multi-partition VTKHDF, connectivity uses LOCAL point indices
  // (each partition's data is concatenated). Do NOT shift point indices.
  // Offsets use LOCAL connectivity indices (restarting from 0 per partition).

  // Extract points as 2D array (numPoints, 3)
  std::vector<double> pointCoords(this->numPoints * 3);
  for (vtkIdType i = 0; i < this->numPoints; i++) {
    double pt[3];
    this->unstructured_grid->GetPoint(i, pt);
    pointCoords[i * 3 + 0] = pt[0];
    pointCoords[i * 3 + 1] = pt[1];
    pointCoords[i * 3 + 2] = pt[2];
  }

  // Create transfer property list (collective for parallel HDF5)
  hid_t dxpl = H5P_DEFAULT;
#ifdef SPECFEM_HDF5_IS_PARALLEL
  if (this->use_parallel_hdf5) {
    dxpl = SPECFEM_H5_CHECK_ID(H5Pcreate(H5P_DATASET_XFER));
    SPECFEM_H5_CHECK(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT));
  }
#endif

  // Write static geometry datasets using helpers
  using specfem::periodic_tasks::plot_wavefield_impl::write_static_1d;
  using specfem::periodic_tasks::plot_wavefield_impl::write_static_2d;

  // Points (2D: numPoints x 3)
  write_static_2d(vtkhdf_group, "Points", H5T_NATIVE_DOUBLE,
                  (hsize_t)this->total_points, 3,
                  (hsize_t)this->global_point_offset, (hsize_t)this->numPoints,
                  pointCoords.data(), this->use_parallel_hdf5, dxpl);

  // Connectivity (1D)
  write_static_1d(vtkhdf_group, "Connectivity", H5T_NATIVE_LLONG,
                  (hsize_t)this->total_connectivity,
                  (hsize_t)this->global_connectivity_offset,
                  (hsize_t)this->numConnectivityIds, connectivity.data(),
                  this->use_parallel_hdf5, dxpl);

  // Offsets (1D): each partition contributes (numCells + 1) entries.
  // Total size = total_cells + num_parts. Offsets are LOCAL per partition.
  {
    hsize_t off_total = (hsize_t)(this->total_cells + this->num_parts);
    hsize_t off_offset = (hsize_t)this->global_cell_offset;
    hsize_t off_count = (hsize_t)(this->numCells + 1);
    if (this->use_parallel_hdf5) {
      // Each prior partition contributes numCells[p] + 1 entries
      off_offset += (hsize_t)specfem::MPI::get_rank();
    }
    write_static_1d(vtkhdf_group, "Offsets", H5T_NATIVE_LLONG, off_total,
                    off_offset, off_count, offsets.data(),
                    this->use_parallel_hdf5, dxpl);
  }

  // Types (1D)
  write_static_1d(vtkhdf_group, "Types", H5T_NATIVE_UCHAR,
                  (hsize_t)this->total_cells, (hsize_t)this->global_cell_offset,
                  (hsize_t)this->numCells, types.data(),
                  this->use_parallel_hdf5, dxpl);

  // Extract and write material IDs as CellData
  const auto &element_types = this->assembly.element_types;
  std::vector<int> material_ids;

  // For Lagrange hexahedral grid: 1 cell per spectral element
  for (int ispec = 0; ispec < this->nspec; ++ispec) {
    const auto material_tag = element_types.get_medium_tag(ispec);
    // Convert enum to integer for HDF5 storage
    const int material_id = static_cast<int>(material_tag);
    material_ids.push_back(material_id);
  }

  // Create CellData group and write material IDs
  hid_t cd_group = SPECFEM_H5_CHECK_ID(H5Gcreate(
      vtkhdf_group, "CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  write_static_1d(cd_group, "MaterialID", H5T_NATIVE_INT,
                  (hsize_t)this->total_cells, (hsize_t)this->global_cell_offset,
                  (hsize_t)material_ids.size(), material_ids.data(),
                  this->use_parallel_hdf5, dxpl);
  SPECFEM_H5_CHECK(H5Gclose(cd_group));

  // Create PointData group
  hid_t pd_group = SPECFEM_H5_CHECK_ID(H5Gcreate(
      vtkhdf_group, "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  // Write static point data: Jacobian
  {
    const auto &h_jacobian = this->assembly.jacobian_matrix.h_jacobian;
    std::vector<float> jacobian_data;
    jacobian_data.reserve(this->numPoints);

    for (int ispec = 0; ispec < this->nspec; ++ispec) {
      for (int iz = 0; iz < this->ngllz; ++iz) {
        for (int iy = 0; iy < this->nglly; ++iy) {
          for (int ix = 0; ix < this->ngllx; ++ix) {

            if (h_jacobian(ispec, iz, iy, ix) < static_cast<type_real>(1e-10)) {
              std::ostringstream err_msg;
              err_msg << "Error: Jacobian is non-positive, invalid element "
                         "mapping.\n";
              err_msg << "ispec: " << ispec << " iz: " << iz << " iy: " << iy
                      << " ix: " << ix
                      << " jacobian: " << h_jacobian(ispec, iz, iy, ix)
                      << std::endl;
              specfem::Logger::error(err_msg.str());
              specfem::program::abort(err_msg.str());
            }
            jacobian_data.push_back(
                static_cast<float>(h_jacobian(ispec, iz, iy, ix)));
          }
        }
      }
    }

    write_static_1d(
        pd_group, "Jacobian", H5T_NATIVE_FLOAT, (hsize_t)this->total_points,
        (hsize_t)this->global_point_offset, (hsize_t)jacobian_data.size(),
        jacobian_data.data(), this->use_parallel_hdf5, dxpl);
  }

  // Create extensible dataset for wavefield scalars
  // Initial size: 0 (will grow as needed)
  hsize_t pd_initial_dims[1] = { 0 };
  hsize_t pd_max_dims[1] = { H5S_UNLIMITED };
  hid_t pd_dataspace =
      SPECFEM_H5_CHECK_ID(H5Screate_simple(1, pd_initial_dims, pd_max_dims));

  // Create dataset creation property list and set chunking
  // Chunk size = one full timestep worth of data (all ranks combined)
  hid_t pd_plist = SPECFEM_H5_CHECK_ID(H5Pcreate(H5P_DATASET_CREATE));
  hsize_t pd_chunk_dims[1] = { (hsize_t)this->total_points };
  SPECFEM_H5_CHECK(H5Pset_chunk(pd_plist, 1, pd_chunk_dims));

  hid_t pd_dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(pd_group, "Wavefield", H5T_NATIVE_FLOAT, pd_dataspace,
                H5P_DEFAULT, pd_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(pd_dataset));
  SPECFEM_H5_CHECK(H5Pclose(pd_plist));
  SPECFEM_H5_CHECK(H5Sclose(pd_dataspace));
  SPECFEM_H5_CHECK(H5Gclose(pd_group));

  // Create extensible temporal metadata arrays instead of pre-allocated ones
  // NumberOfPoints/Cells/ConnectivityIds: num_parts entries per timestep
  hsize_t temp_initial_dims[1] = { 0 };
  hsize_t temp_max_dims[1] = { H5S_UNLIMITED };
  hsize_t temp_chunk_dims[1] = { (hsize_t)this->num_parts };

  // Create dataset creation property list for chunking
  hid_t temp_plist = SPECFEM_H5_CHECK_ID(H5Pcreate(H5P_DATASET_CREATE));
  SPECFEM_H5_CHECK(H5Pset_chunk(temp_plist, 1, temp_chunk_dims));

  hid_t temp_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, temp_initial_dims, temp_max_dims));
  hid_t dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(vtkhdf_group, "NumberOfPoints", H5T_NATIVE_LLONG,
                temp_dataspace, H5P_DEFAULT, temp_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(temp_dataspace));

  temp_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, temp_initial_dims, temp_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(vtkhdf_group, "NumberOfCells", H5T_NATIVE_LLONG, temp_dataspace,
                H5P_DEFAULT, temp_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(temp_dataspace));

  temp_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, temp_initial_dims, temp_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(vtkhdf_group, "NumberOfConnectivityIds", H5T_NATIVE_LLONG,
                temp_dataspace, H5P_DEFAULT, temp_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(temp_dataspace));

  SPECFEM_H5_CHECK(H5Pclose(temp_plist));

  // Create Steps group and extensible metadata
  hid_t steps_group = SPECFEM_H5_CHECK_ID(
      H5Gcreate(vtkhdf_group, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  // Create extensible datasets for time steps metadata
  // NSteps attribute - will be updated during run
  hid_t attr_space = SPECFEM_H5_CHECK_ID(H5Screate(H5S_SCALAR));
  int initial_nsteps = 0;
  hid_t attr =
      SPECFEM_H5_CHECK_ID(H5Acreate(steps_group, "NSteps", H5T_NATIVE_INT,
                                    attr_space, H5P_DEFAULT, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Awrite(attr, H5T_NATIVE_INT, &initial_nsteps));
  SPECFEM_H5_CHECK(H5Aclose(attr));
  SPECFEM_H5_CHECK(H5Sclose(attr_space));

  // Create extensible dataset for time values
  hsize_t steps_initial_dims[1] = { 0 };
  hsize_t steps_max_dims[1] = { H5S_UNLIMITED };
  hsize_t steps_chunk_dims[1] = { 1 };

  hid_t steps_plist = SPECFEM_H5_CHECK_ID(H5Pcreate(H5P_DATASET_CREATE));
  SPECFEM_H5_CHECK(H5Pset_chunk(steps_plist, 1, steps_chunk_dims));

  hid_t steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "Values", H5T_NATIVE_DOUBLE, steps_dataspace,
                H5P_DEFAULT, steps_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  // NumberOfParts: 1 entry per timestep (like Values), uses steps_plist
  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "NumberOfParts", H5T_NATIVE_LLONG, steps_dataspace,
                H5P_DEFAULT, steps_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  // Offset arrays: 1 entry per timestep (these index into the geometry/data)
  hid_t offsets_plist = SPECFEM_H5_CHECK_ID(H5Pcreate(H5P_DATASET_CREATE));
  hsize_t offsets_chunk_dims[1] = { 1 };
  SPECFEM_H5_CHECK(H5Pset_chunk(offsets_plist, 1, offsets_chunk_dims));

  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "PartOffsets", H5T_NATIVE_LLONG, steps_dataspace,
                H5P_DEFAULT, offsets_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "PointOffsets", H5T_NATIVE_LLONG, steps_dataspace,
                H5P_DEFAULT, offsets_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "CellOffsets", H5T_NATIVE_LLONG, steps_dataspace,
                H5P_DEFAULT, offsets_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "ConnectivityIdOffsets", H5T_NATIVE_LLONG,
                steps_dataspace, H5P_DEFAULT, offsets_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  // Create PointDataOffsets subgroup with extensible datasets
  hid_t pd_offsets_group = SPECFEM_H5_CHECK_ID(H5Gcreate(
      steps_group, "PointDataOffsets", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  // Wavefield offsets: 1 entry per timestep
  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(pd_offsets_group, "Wavefield", H5T_NATIVE_LLONG,
                steps_dataspace, H5P_DEFAULT, offsets_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));
  SPECFEM_H5_CHECK(H5Gclose(pd_offsets_group));

  SPECFEM_H5_CHECK(H5Pclose(offsets_plist));
  SPECFEM_H5_CHECK(H5Pclose(steps_plist));
  SPECFEM_H5_CHECK(H5Gclose(steps_group));

  // Close transfer property list if created
#ifdef SPECFEM_HDF5_IS_PARALLEL
  if (this->use_parallel_hdf5) {
    SPECFEM_H5_CHECK(H5Pclose(dxpl));
  }
#endif

  // Close HDF5 file - will reopen for each timestep write
  SPECFEM_H5_CHECK(H5Gclose(vtkhdf_group));
  SPECFEM_H5_CHECK(H5Fclose(hdf5_file_id));

  specfem::Logger::info([&](std::ostringstream &oss) {
    oss << "Initialized VTK HDF5 file for 3D wavefield output: "
        << this->hdf5_filename << " (" << this->num_parts << " part(s), "
        << this->total_points << " total points)";
  });

#else
  throw std::runtime_error(
      "VTK HDF5 output requested but HDF5 support not compiled.");
#endif
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::initialize(
        specfem::assembly::assembly<dimension_tag> &assembly) {

  // Create the grid structure
  create_lagrange_hex_grid();

  // Compute initial wavefield scalars and add to grid
  auto scalars = compute_wavefield_scalars(assembly);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  switch (output_format) {
  case specfem::enums::display_format::vtkhdf:
    this->initialize_vtkhdf(scalars);
    break;
  default:
    throw std::runtime_error("Unsupported display format for 3D");
  }
}

#ifndef NO_HDF5
// Helper: extend a 1D dataset, select hyperslab at write_offset, write one
// scalar value, then close the dataset.
// When do_write is false, the dataset is extended (collective in parallel HDF5)
// but no data is written (for non-writing ranks).
void specfem::periodic_tasks::plot_wavefield<
    specfem::element::dimension_tag::dim3>::
    extend_and_write_scalar(hid_t parent, const char *dataset_name,
                            hsize_t new_extent, hsize_t write_offset,
                            hid_t mem_type, const void *data, hid_t dxpl,
                            bool do_write) {
  hid_t ds = SPECFEM_H5_CHECK_ID(H5Dopen(parent, dataset_name, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dset_extent(ds, &new_extent));

  if (do_write) {
    hid_t filespace = SPECFEM_H5_CHECK_ID(H5Dget_space(ds));
    hsize_t count = 1;
    SPECFEM_H5_CHECK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                         &write_offset, NULL, &count, NULL));
    hid_t memspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, &count, NULL));
    SPECFEM_H5_CHECK(H5Dwrite(ds, mem_type, memspace, filespace, dxpl, data));
    SPECFEM_H5_CHECK(H5Sclose(memspace));
    SPECFEM_H5_CHECK(H5Sclose(filespace));
  }

  SPECFEM_H5_CHECK(H5Dclose(ds));
}

// Helper: extend a 1D dataset, select hyperslab at write_offset with the given
// count, write an array of values, then close the dataset.
// When do_write is false, the dataset is extended but no data is written.
void specfem::periodic_tasks::plot_wavefield<
    specfem::element::dimension_tag::dim3>::
    extend_and_write_array(hid_t parent, const char *dataset_name,
                           hsize_t new_extent, hsize_t write_offset,
                           hsize_t count, hid_t mem_type, const void *data,
                           hid_t dxpl, bool do_write) {
  hid_t ds = SPECFEM_H5_CHECK_ID(H5Dopen(parent, dataset_name, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dset_extent(ds, &new_extent));

  if (do_write) {
    hid_t filespace = SPECFEM_H5_CHECK_ID(H5Dget_space(ds));
    SPECFEM_H5_CHECK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                         &write_offset, NULL, &count, NULL));
    hid_t memspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, &count, NULL));
    SPECFEM_H5_CHECK(H5Dwrite(ds, mem_type, memspace, filespace, dxpl, data));
    SPECFEM_H5_CHECK(H5Sclose(memspace));
    SPECFEM_H5_CHECK(H5Sclose(filespace));
  }

  SPECFEM_H5_CHECK(H5Dclose(ds));
}
#endif // NO_HDF5

#ifndef NO_HDF5
void specfem::periodic_tasks::plot_wavefield<
    specfem::element::dimension_tag::dim3>::compute_mpi_offsets() {
#ifdef SPECFEM_ENABLE_MPI
  if (specfem::MPI::get_size() > 1 && this->use_parallel_hdf5) {
    MPI_Comm comm = specfem::MPI::communicator();

    SPECFEM_MPI_SAFECALL(MPI_Exscan(&this->numPoints,
                                    &this->global_point_offset, 1,
                                    MPI_LONG_LONG, MPI_SUM, comm));
    SPECFEM_MPI_SAFECALL(MPI_Exscan(&this->numCells, &this->global_cell_offset,
                                    1, MPI_LONG_LONG, MPI_SUM, comm));
    SPECFEM_MPI_SAFECALL(MPI_Exscan(&this->numConnectivityIds,
                                    &this->global_connectivity_offset, 1,
                                    MPI_LONG_LONG, MPI_SUM, comm));

    // MPI_Exscan leaves rank 0 undefined — set to 0 explicitly
    if (specfem::MPI::get_rank() == 0) {
      this->global_point_offset = 0;
      this->global_cell_offset = 0;
      this->global_connectivity_offset = 0;
    }

    SPECFEM_MPI_SAFECALL(MPI_Allreduce(&this->numPoints, &this->total_points, 1,
                                       MPI_LONG_LONG, MPI_SUM, comm));
    SPECFEM_MPI_SAFECALL(MPI_Allreduce(&this->numCells, &this->total_cells, 1,
                                       MPI_LONG_LONG, MPI_SUM, comm));
    SPECFEM_MPI_SAFECALL(MPI_Allreduce(&this->numConnectivityIds,
                                       &this->total_connectivity, 1,
                                       MPI_LONG_LONG, MPI_SUM, comm));

    this->num_parts = specfem::MPI::get_size();

    // Rank 0 gathers all point offsets (needed for PointDataOffsets)
    this->all_point_offsets.resize(this->num_parts);
    SPECFEM_MPI_SAFECALL(
        MPI_Gather(&this->global_point_offset, 1, MPI_LONG_LONG,
                   this->all_point_offsets.data(), 1, MPI_LONG_LONG, 0, comm));

    // Rank 0 gathers per-rank counts (needed for NumberOfPoints/Cells/Conn)
    this->all_point_counts.resize(this->num_parts);
    this->all_cell_counts.resize(this->num_parts);
    this->all_connectivity_counts.resize(this->num_parts);
    SPECFEM_MPI_SAFECALL(MPI_Gather(&this->numPoints, 1, MPI_LONG_LONG,
                                    this->all_point_counts.data(), 1,
                                    MPI_LONG_LONG, 0, comm));
    SPECFEM_MPI_SAFECALL(MPI_Gather(&this->numCells, 1, MPI_LONG_LONG,
                                    this->all_cell_counts.data(), 1,
                                    MPI_LONG_LONG, 0, comm));
    SPECFEM_MPI_SAFECALL(MPI_Gather(&this->numConnectivityIds, 1, MPI_LONG_LONG,
                                    this->all_connectivity_counts.data(), 1,
                                    MPI_LONG_LONG, 0, comm));
    return;
  }
#endif // SPECFEM_ENABLE_MPI

  // Serial path: global == local
  this->global_point_offset = 0;
  this->global_cell_offset = 0;
  this->global_connectivity_offset = 0;
  this->total_points = this->numPoints;
  this->total_cells = this->numCells;
  this->total_connectivity = this->numConnectivityIds;
  this->num_parts = 1;
  this->all_point_offsets = { 0 };
  this->all_point_counts = { this->numPoints };
  this->all_cell_counts = { this->numCells };
  this->all_connectivity_counts = { this->numConnectivityIds };
}
#endif // NO_HDF5

#ifndef NO_HDF5
hid_t specfem::periodic_tasks::plot_wavefield<
    specfem::element::dimension_tag::dim3>::create_file_access_plist() const {
  hid_t fapl = SPECFEM_H5_CHECK_ID(H5Pcreate(H5P_FILE_ACCESS));
#ifdef SPECFEM_HDF5_IS_PARALLEL
  if (this->use_parallel_hdf5) {
    SPECFEM_H5_CHECK(
        H5Pset_fapl_mpio(fapl, specfem::MPI::communicator(), MPI_INFO_NULL));
  }
#endif // SPECFEM_HDF5_IS_PARALLEL
  return fapl;
}
#endif // NO_HDF5

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::run_vtkhdf(
        vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {

#ifndef NO_HDF5
  // Open HDF5 file for extending datasets
  hid_t fapl = this->create_file_access_plist();
  hid_t hdf5_file_id = SPECFEM_H5_CHECK_ID(
      H5Fopen(this->hdf5_filename.c_str(), H5F_ACC_RDWR, fapl));
  SPECFEM_H5_CHECK(H5Pclose(fapl));
  hid_t vtkhdf_group =
      SPECFEM_H5_CHECK_ID(H5Gopen(hdf5_file_id, "/VTKHDF", H5P_DEFAULT));

  // Create transfer property list for parallel HDF5.
  // Use independent I/O — each rank writes its own hyperslab independently.
  // Collective I/O can be enabled later as an optimization once correctness
  // is verified.
  hid_t dxpl = H5P_DEFAULT;
#ifdef SPECFEM_HDF5_IS_PARALLEL
  if (this->use_parallel_hdf5) {
    dxpl = SPECFEM_H5_CHECK_ID(H5Pcreate(H5P_DATASET_XFER));
    SPECFEM_H5_CHECK(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT));
  }
#endif

  // Extend and write wavefield data
  hid_t pd_group =
      SPECFEM_H5_CHECK_ID(H5Gopen(vtkhdf_group, "PointData", H5P_DEFAULT));
  hid_t pd_dataset =
      SPECFEM_H5_CHECK_ID(H5Dopen(pd_group, "Wavefield", H5P_DEFAULT));

  // Extend the wavefield dataset to accommodate new timestep
  hsize_t new_size[1] = { (hsize_t)((this->current_timestep + 1) *
                                    this->total_points) };
  SPECFEM_H5_CHECK(H5Dset_extent(pd_dataset, new_size));

  // Write wavefield data for this timestep
  std::vector<float> scalar_data(this->numPoints);
  for (long long i = 0; i < this->numPoints; i++) {
    scalar_data[i] = scalars->GetValue(i);
  }

  // Calculate offset and count for this timestep
  // For parallel: each rank writes at its global offset within this timestep
  hsize_t offset = (hsize_t)(this->current_timestep * this->total_points +
                             this->global_point_offset);
  hsize_t count = (hsize_t)this->numPoints;

  // Select hyperslab in the file dataset
  hid_t filespace = SPECFEM_H5_CHECK_ID(H5Dget_space(pd_dataset));
  SPECFEM_H5_CHECK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL,
                                       &count, NULL));

  // Create memory dataspace and write
  hid_t memspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, &count, NULL));
  SPECFEM_H5_CHECK(H5Dwrite(pd_dataset, H5T_NATIVE_FLOAT, memspace, filespace,
                            dxpl, scalar_data.data()));

  SPECFEM_H5_CHECK(H5Sclose(memspace));
  SPECFEM_H5_CHECK(H5Sclose(filespace));
  SPECFEM_H5_CHECK(H5Dclose(pd_dataset));
  SPECFEM_H5_CHECK(H5Gclose(pd_group));

  // Update temporal metadata arrays (use global totals).
  // In parallel HDF5, H5Dset_extent is collective — ALL ranks must call it.
  // Only rank 0 actually writes data; other ranks extend but select H5S_NONE.
  bool is_metadata_writer =
      !this->use_parallel_hdf5 || specfem::MPI::main_proc();

  hsize_t new_ts_count = (hsize_t)(this->current_timestep + 1);
  hsize_t ts_offset = (hsize_t)this->current_timestep;

  // NumberOfPoints/Cells/ConnectivityIds: num_parts entries total (static
  // geometry — partition counts are written once and reused every timestep).
  // PartOffsets[t] = 0 tells VTK to always read counts from index 0.
  if (this->current_timestep == 0) {
    hsize_t count_extent = (hsize_t)this->num_parts;
    hsize_t count_offset = 0;
    hsize_t count_count = (hsize_t)this->num_parts;

    extend_and_write_array(vtkhdf_group, "NumberOfPoints", count_extent,
                           count_offset, count_count, H5T_NATIVE_LLONG,
                           this->all_point_counts.data(), dxpl,
                           is_metadata_writer);
    extend_and_write_array(vtkhdf_group, "NumberOfCells", count_extent,
                           count_offset, count_count, H5T_NATIVE_LLONG,
                           this->all_cell_counts.data(), dxpl,
                           is_metadata_writer);
    extend_and_write_array(
        vtkhdf_group, "NumberOfConnectivityIds", count_extent, count_offset,
        count_count, H5T_NATIVE_LLONG, this->all_connectivity_counts.data(),
        dxpl, is_metadata_writer);
  }

  // Update Steps metadata
  hid_t steps_group =
      SPECFEM_H5_CHECK_ID(H5Gopen(vtkhdf_group, "Steps", H5P_DEFAULT));

  {
    double time_value =
        static_cast<double>(this->assembly.t0) +
        static_cast<double>(istep) * static_cast<double>(this->assembly.dt);
    extend_and_write_scalar(steps_group, "Values", new_ts_count, ts_offset,
                            H5T_NATIVE_DOUBLE, &time_value, dxpl,
                            is_metadata_writer);

    long long num_parts_val = this->num_parts;
    extend_and_write_scalar(steps_group, "NumberOfParts", new_ts_count,
                            ts_offset, H5T_NATIVE_LLONG, &num_parts_val, dxpl,
                            is_metadata_writer);

    // Offset arrays: 1 entry per timestep.
    // Static geometry: PartOffsets = 0 (always read counts from index 0
    // in NumberOfCells/Points/ConnectivityIds). Geometry offsets = 0
    // (Points/Cells/Connectivity don't change between timesteps).
    long long zero = 0;
    extend_and_write_scalar(steps_group, "PartOffsets", new_ts_count, ts_offset,
                            H5T_NATIVE_LLONG, &zero, dxpl, is_metadata_writer);

    extend_and_write_scalar(steps_group, "PointOffsets", new_ts_count,
                            ts_offset, H5T_NATIVE_LLONG, &zero, dxpl,
                            is_metadata_writer);
    extend_and_write_scalar(steps_group, "CellOffsets", new_ts_count, ts_offset,
                            H5T_NATIVE_LLONG, &zero, dxpl, is_metadata_writer);
    extend_and_write_scalar(steps_group, "ConnectivityIdOffsets", new_ts_count,
                            ts_offset, H5T_NATIVE_LLONG, &zero, dxpl,
                            is_metadata_writer);

    // PointDataOffsets/Wavefield: 1 entry per timestep
    // Offset into the Wavefield dataset for this timestep's data
    hid_t pd_offsets_group = SPECFEM_H5_CHECK_ID(
        H5Gopen(steps_group, "PointDataOffsets", H5P_DEFAULT));

    long long wf_offset_val =
        (long long)(this->current_timestep * this->total_points);
    extend_and_write_scalar(pd_offsets_group, "Wavefield", new_ts_count,
                            ts_offset, H5T_NATIVE_LLONG, &wf_offset_val, dxpl,
                            is_metadata_writer);
    SPECFEM_H5_CHECK(H5Gclose(pd_offsets_group));

    // Update NSteps attribute — all ranks write the same value.
    // In parallel HDF5, attribute writes are not collective but modifying
    // metadata from a single rank can cause synchronization issues at close.
    {
      int nsteps_written = this->current_timestep + 1;
      hid_t attr =
          SPECFEM_H5_CHECK_ID(H5Aopen(steps_group, "NSteps", H5P_DEFAULT));
      SPECFEM_H5_CHECK(H5Awrite(attr, H5T_NATIVE_INT, &nsteps_written));
      SPECFEM_H5_CHECK(H5Aclose(attr));
    }
  }

  SPECFEM_H5_CHECK(H5Gclose(steps_group));

  // Close transfer property list if created
#ifdef SPECFEM_HDF5_IS_PARALLEL
  if (this->use_parallel_hdf5) {
    SPECFEM_H5_CHECK(H5Pclose(dxpl));
  }
#endif

  // Close HDF5 resources
  SPECFEM_H5_CHECK(H5Gclose(vtkhdf_group));

  // Flush all buffers before close — ensures metadata is synchronized
  // across ranks, preventing H5Fclose deadlocks in parallel HDF5.
  SPECFEM_H5_CHECK(H5Fflush(hdf5_file_id, H5F_SCOPE_GLOBAL));

  SPECFEM_H5_CHECK(H5Fclose(hdf5_file_id));

  this->current_timestep++;

  specfem::Logger::info([&](std::ostringstream &oss) {
    oss << "Wrote 3D wavefield data for timestep " << istep
        << " to HDF5 file (step " << this->current_timestep << ")";
  });

#else
  throw std::runtime_error(
      "VTK HDF5 output requested but HDF5 support not compiled.");
#endif
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::run(
        specfem::assembly::assembly<dimension_tag> &assembly, const int istep) {

  // Update the wavefield scalars only
  auto scalars = compute_wavefield_scalars(assembly);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  switch (output_format) {
  case specfem::enums::display_format::vtkhdf:
    this->run_vtkhdf(scalars, istep);
    break;
  default:
    throw std::runtime_error("Unsupported output format for 3D");
  }
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::finalize(
        specfem::assembly::assembly<dimension_tag> &assembly) {

  switch (output_format) {
  case specfem::enums::display_format::vtkhdf:
    // Clean up VTK objects
    unstructured_grid = nullptr;
    break;
  default:
    throw std::runtime_error("Unsupported output format for 3D");
  }
}

#endif // NO_VTK
