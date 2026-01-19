#include "plot_wavefield.hpp"
#include "enumerations/display.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/logger.hpp"
#include "specfem/periodic_tasks/plotter.hpp"
#include "specfem/program.hpp"
#include "specfem/utilities.hpp"

#ifdef NO_VTK

#include <sstream>

#else

#include <algorithm>
#include <boost/filesystem.hpp>
#include <cmath>
#include <fstream>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkLagrangeHexahedron.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

#ifndef NO_HDF5
#include <hdf5.h>
#endif // NO_HDF5

#endif // NO_VTK

#ifdef NO_VTK

// Add constructor implementation for NO_VTK builds
specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    plot_wavefield(
        const specfem::assembly::assembly<dimension_tag> &assembly,
        const specfem::display::format &output_format,
        const specfem::wavefield::type &wavefield_type,
        const specfem::wavefield::simulation_field &simulation_wavefield_type,
        const specfem::display::component &component, const type_real &dt,
        const int &time_interval, const boost::filesystem::path &output_folder)
    : assembly(assembly), simulation_wavefield_type(simulation_wavefield_type),
      wavefield_type(wavefield_type), component(component),
      plotter<dimension_tag>(time_interval), output_format(output_format),
      output_folder(output_folder), nspec(assembly.mesh.nspec), dt(dt),
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

void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    run(specfem::assembly::assembly<dimension_tag> &assembly, const int istep) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    initialize(specfem::assembly::assembly<dimension_tag> &assembly) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    finalize(specfem::assembly::assembly<dimension_tag> &assembly) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

#else

// Constructor
specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    plot_wavefield(
        const specfem::assembly::assembly<dimension_tag> &assembly,
        const specfem::display::format &output_format,
        const specfem::wavefield::type &wavefield_type,
        const specfem::wavefield::simulation_field &simulation_wavefield_type,
        const specfem::display::component &component, const type_real &dt,
        const int &time_interval, const boost::filesystem::path &output_folder)
    : assembly(assembly), simulation_wavefield_type(simulation_wavefield_type),
      wavefield_type(wavefield_type), component(component),
      plotter<dimension_tag>(time_interval), output_format(output_format),
      output_folder(output_folder), nspec(assembly.mesh.nspec), dt(dt),
      ngllx(assembly.mesh.element_grid.ngllx),
      nglly(assembly.mesh.element_grid.nglly),
      ngllz(assembly.mesh.element_grid.ngllz) {
  // Only VTK HDF5 output is supported for 3D
  if (output_format != specfem::display::format::vtkhdf) {
    throw std::runtime_error(
        "Only VTK HDF5 output format is supported for 3D wavefield plotting");
  }
};

// Get wavefield type to display
specfem::wavefield::type specfem::periodic_tasks::plot_wavefield<
    specfem::dimension::type::dim3>::get_wavefield_type() {
  if (wavefield_type == specfem::wavefield::type::displacement) {
    return specfem::wavefield::type::displacement;
  } else if (wavefield_type == specfem::wavefield::type::velocity) {
    return specfem::wavefield::type::velocity;
  } else if (wavefield_type == specfem::wavefield::type::acceleration) {
    return specfem::wavefield::type::acceleration;
  } else {
    throw std::runtime_error("Wavefield type not supported");
  }
}

// Helper function to get scalar value at a given point
float specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    get_scalar_value_at_point(
        const Kokkos::View<type_real *****, Kokkos::LayoutLeft,
                           Kokkos::HostSpace> &wavefield_data,
        const specfem::wavefield::type &wavefield_type,
        const specfem::display::component &component, const int ispec,
        const int iz, const int iy, const int ix) {

  if (wavefield_type == specfem::wavefield::type::pressure ||
      wavefield_type == specfem::wavefield::type::rotation ||
      wavefield_type == specfem::wavefield::type::intrinsic_rotation ||
      wavefield_type == specfem::wavefield::type::curl) {
    return std::abs(wavefield_data(ispec, iz, iy, ix, 0));
  }

  // Computing the component or magnitude for vector fields
  if (component == specfem::display::component::x) {
    return wavefield_data(ispec, iz, iy, ix, 0);
  } else if (component == specfem::display::component::y) {
    return wavefield_data(ispec, iz, iy, ix, 1);
  } else if (component == specfem::display::component::z) {
    return wavefield_data(ispec, iz, iy, ix, 2);
  } else if (component == specfem::display::component::magnitude) {
    // Compute magnitude from 3-component vector
    type_real magnitude = 0.0;
    for (int icomp = 0; icomp < 3; ++icomp) {
      const type_real comp = wavefield_data(ispec, iz, iy, ix, icomp);
      magnitude += comp * comp;
    }
    return static_cast<float>(std::sqrt(magnitude));
  } else {
    throw std::runtime_error("Invalid component,'" +
                             specfem::display::to_string(component) +
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
    specfem::dimension::type::dim3>::create_lagrange_hex_grid() {
  const auto &coordinates = assembly.mesh.h_coord;

  // Each spectral element becomes one Lagrange hexahedron
  const int ncells = nspec;
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
vtkSmartPointer<vtkFloatArray> specfem::periodic_tasks::
    plot_wavefield<specfem::dimension::type::dim3>::compute_wavefield_scalars(
        specfem::assembly::assembly<dimension_tag> &assembly) {
  const auto wavefield_type = get_wavefield_type();
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

template <>
void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    initialize<specfem::display::format::vtkhdf>(
        vtkSmartPointer<vtkFloatArray> &scalars) {

#ifndef NO_HDF5

  // Initialize VTK HDF5 file for time series output
  this->current_timestep = 0;
  this->numPoints = 0;
  this->numCells = 0;

  // Create HDF5 file
  this->hdf5_filename = (this->output_folder / "wavefield.vtkhdf").string();
  hid_t hdf5_file_id = H5Fcreate(this->hdf5_filename.c_str(), H5F_ACC_TRUNC,
                                 H5P_DEFAULT, H5P_DEFAULT);
  hid_t vtkhdf_group =
      H5Gcreate(hdf5_file_id, "/VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Set VTKHDF attributes
  {
    hid_t attr_space, attr;
    int version[2] = { 2, 0 };
    hsize_t dims[1] = { 2 };
    attr_space = H5Screate_simple(1, dims, NULL);
    attr = H5Acreate(vtkhdf_group, "Version", H5T_NATIVE_INT, attr_space,
                     H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, version);
    H5Aclose(attr);
    H5Sclose(attr_space);

    // Set Type attribute
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, 16);
    attr_space = H5Screate(H5S_SCALAR);
    attr = H5Acreate(vtkhdf_group, "Type", str_type, attr_space, H5P_DEFAULT,
                     H5P_DEFAULT);
    const char *type_str = "UnstructuredGrid";
    H5Awrite(attr, str_type, type_str);
    H5Aclose(attr);
    H5Sclose(attr_space);
    H5Tclose(str_type);
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

  // Extract points as 2D array (numPoints, 3)
  std::vector<double> pointCoords(this->numPoints * 3);
  for (vtkIdType i = 0; i < this->numPoints; i++) {
    double pt[3];
    this->unstructured_grid->GetPoint(i, pt);
    pointCoords[i * 3 + 0] = pt[0];
    pointCoords[i * 3 + 1] = pt[1];
    pointCoords[i * 3 + 2] = pt[2];
  }

  // Write Points (static geometry) - 2D array (numPoints, 3)
  hsize_t point_dims[2] = { (hsize_t)this->numPoints, 3 };
  hid_t dataspace = H5Screate_simple(2, point_dims, NULL);
  hid_t dataset = H5Dcreate(vtkhdf_group, "Points", H5T_NATIVE_DOUBLE,
                            dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           pointCoords.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);

  // Write Connectivity (static)
  hsize_t dims[1];
  dims[0] = connectivity.size();
  dataspace = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(vtkhdf_group, "Connectivity", H5T_NATIVE_LLONG, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           connectivity.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);

  // Write Offsets (static)
  dims[0] = offsets.size();
  dataspace = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(vtkhdf_group, "Offsets", H5T_NATIVE_LLONG, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           offsets.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);

  // Write Types (static)
  dims[0] = types.size();
  dataspace = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(vtkhdf_group, "Types", H5T_NATIVE_UCHAR, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           types.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);

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
  hid_t cd_group = H5Gcreate(vtkhdf_group, "CellData", H5P_DEFAULT, H5P_DEFAULT,
                             H5P_DEFAULT);
  dims[0] = material_ids.size();
  dataspace = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(cd_group, "MaterialID", H5T_NATIVE_INT, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           material_ids.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Gclose(cd_group);

  // Create PointData group
  hid_t pd_group = H5Gcreate(vtkhdf_group, "PointData", H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);

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

    dims[0] = jacobian_data.size();
    dataspace = H5Screate_simple(1, dims, NULL);
    dataset = H5Dcreate(pd_group, "Jacobian", H5T_NATIVE_FLOAT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             jacobian_data.data());
    H5Dclose(dataset);
    H5Sclose(dataspace);
  }

  // Write static point data: Material properties (kappa, mu, rho)
  // Note: These are only available for elastic isotropic materials
  {
    std::vector<float> kappa_data, mu_data, rho_data;
    kappa_data.reserve(this->numPoints);
    mu_data.reserve(this->numPoints);
    rho_data.reserve(this->numPoints);

    // Get the elastic isotropic properties container
    const auto &elastic_properties =
        this->assembly.properties
            .get_container<specfem::element::medium_tag::elastic,
                           specfem::element::property_tag::isotropic>();

    // Access the properties through the assembly
    // We need to loop through each point and get the properties
    for (int ispec = 0; ispec < this->nspec; ++ispec) {
      const auto medium_tag =
          this->assembly.element_types.get_medium_tag(ispec);
      const auto property_tag =
          this->assembly.element_types.get_property_tag(ispec);

      // Check if this is elastic isotropic material
      if (medium_tag == specfem::element::medium_tag::elastic &&
          property_tag == specfem::element::property_tag::isotropic) {
        // Get the property index for this element
        const int property_index =
            this->assembly.properties.h_property_index_mapping(ispec);

        for (int iz = 0; iz < this->ngllz; ++iz) {
          for (int iy = 0; iy < this->nglly; ++iy) {
            for (int ix = 0; ix < this->ngllx; ++ix) {
              kappa_data.push_back(static_cast<float>(
                  elastic_properties.h_kappa(property_index, iz, iy, ix)));
              mu_data.push_back(static_cast<float>(
                  elastic_properties.h_mu(property_index, iz, iy, ix)));
              rho_data.push_back(static_cast<float>(
                  elastic_properties.h_rho(property_index, iz, iy, ix)));
            }
          }
        }
      } else {
        // For non-elastic-isotropic materials, write zeros or NaN
        for (int iz = 0; iz < this->ngllz; ++iz) {
          for (int iy = 0; iy < this->nglly; ++iy) {
            for (int ix = 0; ix < this->ngllx; ++ix) {
              kappa_data.push_back(0.0f);
              mu_data.push_back(0.0f);
              rho_data.push_back(0.0f);
            }
          }
        }
      }
    }

    // Write kappa
    dims[0] = kappa_data.size();
    dataspace = H5Screate_simple(1, dims, NULL);
    dataset = H5Dcreate(pd_group, "Kappa", H5T_NATIVE_FLOAT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             kappa_data.data());
    H5Dclose(dataset);
    H5Sclose(dataspace);

    // Write mu
    dims[0] = mu_data.size();
    dataspace = H5Screate_simple(1, dims, NULL);
    dataset = H5Dcreate(pd_group, "Mu", H5T_NATIVE_FLOAT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             mu_data.data());
    H5Dclose(dataset);
    H5Sclose(dataspace);

    // Write rho
    dims[0] = rho_data.size();
    dataspace = H5Screate_simple(1, dims, NULL);
    dataset = H5Dcreate(pd_group, "Rho", H5T_NATIVE_FLOAT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             rho_data.data());
    H5Dclose(dataset);
    H5Sclose(dataspace);
  }

  // Create extensible dataset for wavefield scalars
  // Initial size: 0 (will grow as needed)
  hsize_t pd_initial_dims[1] = { 0 };
  hsize_t pd_max_dims[1] = { H5S_UNLIMITED };
  hid_t pd_dataspace = H5Screate_simple(1, pd_initial_dims, pd_max_dims);

  // Create dataset creation property list and set chunking
  hid_t pd_plist = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t pd_chunk_dims[1] = {
    (hsize_t)this->numPoints
  }; // Chunk size = one timestep worth of data
  H5Pset_chunk(pd_plist, 1, pd_chunk_dims);

  hid_t pd_dataset =
      H5Dcreate(pd_group, "Wavefield", H5T_NATIVE_FLOAT, pd_dataspace,
                H5P_DEFAULT, pd_plist, H5P_DEFAULT);
  H5Dclose(pd_dataset);
  H5Pclose(pd_plist);
  H5Sclose(pd_dataspace);
  H5Gclose(pd_group);

  // Create extensible temporal metadata arrays instead of pre-allocated ones
  hsize_t temp_initial_dims[1] = { 0 };
  hsize_t temp_max_dims[1] = { H5S_UNLIMITED };
  hsize_t temp_chunk_dims[1] = { 1 }; // Chunk size = 1 timestep

  // Create dataset creation property list for chunking
  hid_t temp_plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(temp_plist, 1, temp_chunk_dims);

  hid_t temp_dataspace = H5Screate_simple(1, temp_initial_dims, temp_max_dims);
  dataset = H5Dcreate(vtkhdf_group, "NumberOfPoints", H5T_NATIVE_LLONG,
                      temp_dataspace, H5P_DEFAULT, temp_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(temp_dataspace);

  temp_dataspace = H5Screate_simple(1, temp_initial_dims, temp_max_dims);
  dataset = H5Dcreate(vtkhdf_group, "NumberOfCells", H5T_NATIVE_LLONG,
                      temp_dataspace, H5P_DEFAULT, temp_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(temp_dataspace);

  temp_dataspace = H5Screate_simple(1, temp_initial_dims, temp_max_dims);
  dataset = H5Dcreate(vtkhdf_group, "NumberOfConnectivityIds", H5T_NATIVE_LLONG,
                      temp_dataspace, H5P_DEFAULT, temp_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(temp_dataspace);

  H5Pclose(temp_plist);

  // Create Steps group and extensible metadata
  hid_t steps_group =
      H5Gcreate(vtkhdf_group, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Create extensible datasets for time steps metadata
  // NSteps attribute - will be updated during run
  hid_t attr_space = H5Screate(H5S_SCALAR);
  int initial_nsteps = 0;
  hid_t attr = H5Acreate(steps_group, "NSteps", H5T_NATIVE_INT, attr_space,
                         H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, H5T_NATIVE_INT, &initial_nsteps);
  H5Aclose(attr);
  H5Sclose(attr_space);

  // Create extensible dataset for time values
  hsize_t steps_initial_dims[1] = { 0 };
  hsize_t steps_max_dims[1] = { H5S_UNLIMITED };
  hsize_t steps_chunk_dims[1] = { 1 };

  hid_t steps_plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(steps_plist, 1, steps_chunk_dims);

  hid_t steps_dataspace =
      H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "Values", H5T_NATIVE_DOUBLE, steps_dataspace,
                      H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  // Create extensible datasets for NumberOfParts and offset arrays
  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "NumberOfParts", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "PartOffsets", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "PointOffsets", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "CellOffsets", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "ConnectivityIdOffsets", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  // Create PointDataOffsets subgroup with extensible datasets
  hid_t pd_offsets_group = H5Gcreate(steps_group, "PointDataOffsets",
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Wavefield offsets (extensible)
  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(pd_offsets_group, "Wavefield", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);
  H5Gclose(pd_offsets_group);

  H5Pclose(steps_plist);
  H5Gclose(steps_group);

  // Close HDF5 file - will reopen for each timestep write
  H5Gclose(vtkhdf_group);
  H5Fclose(hdf5_file_id);

  specfem::Logger::info("Initialized VTK HDF5 file for 3D wavefield output: " +
                        this->hdf5_filename + " (extensible datasets)");

#else
  throw std::runtime_error(
      "VTK HDF5 output requested but HDF5 support not compiled.");
#endif
}

void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    initialize_display(vtkSmartPointer<vtkFloatArray> &scalars) {
  // Not implemented for 3D - only HDF5 output is supported
  throw std::runtime_error(
      "Display initialization not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    initialize<specfem::display::format::on_screen>(
        vtkSmartPointer<vtkFloatArray> &scalars) {
  throw std::runtime_error(
      "On-screen display not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    initialize<specfem::display::format::PNG>(
        vtkSmartPointer<vtkFloatArray> &scalars) {
  throw std::runtime_error(
      "PNG output not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    initialize<specfem::display::format::JPG>(
        vtkSmartPointer<vtkFloatArray> &scalars) {
  throw std::runtime_error(
      "JPG output not supported for 3D. Use VTK HDF5 output.");
}

void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    initialize(specfem::assembly::assembly<dimension_tag> &assembly) {

  // Create the grid structure
  create_lagrange_hex_grid();

  // Compute initial wavefield scalars and add to grid
  auto scalars = compute_wavefield_scalars(assembly);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  switch (output_format) {
  case specfem::display::format::vtkhdf:
    this->initialize<specfem::display::format::vtkhdf>(scalars);
    break;
  case specfem::display::format::on_screen:
    this->initialize<specfem::display::format::on_screen>(scalars);
    break;
  case specfem::display::format::PNG:
    this->initialize<specfem::display::format::PNG>(scalars);
    break;
  case specfem::display::format::JPG:
    this->initialize<specfem::display::format::JPG>(scalars);
    break;
  default:
    throw std::runtime_error("Unsupported display format for 3D");
  }

  return;
}

void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    run_render(vtkSmartPointer<vtkFloatArray> &scalars) {
  // Not implemented for 3D
  throw std::runtime_error(
      "Rendering not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    run<specfem::display::format::on_screen>(
        vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {
  throw std::runtime_error(
      "On-screen display not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    run<specfem::display::format::PNG>(vtkSmartPointer<vtkFloatArray> &scalars,
                                       const int istep) {
  throw std::runtime_error(
      "PNG output not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    run<specfem::display::format::JPG>(vtkSmartPointer<vtkFloatArray> &scalars,
                                       const int istep) {
  throw std::runtime_error(
      "JPG output not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    run<specfem::display::format::vtkhdf>(
        vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {

#ifndef NO_HDF5
  // Open HDF5 file for extending datasets
  hid_t hdf5_file_id =
      H5Fopen(this->hdf5_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t vtkhdf_group = H5Gopen(hdf5_file_id, "/VTKHDF", H5P_DEFAULT);

  // Extend and write wavefield data
  hid_t pd_group = H5Gopen(vtkhdf_group, "PointData", H5P_DEFAULT);
  hid_t pd_dataset = H5Dopen(pd_group, "Wavefield", H5P_DEFAULT);

  // Extend the wavefield dataset to accommodate new timestep
  hsize_t new_size[1] = { (hsize_t)((this->current_timestep + 1) *
                                    this->numPoints) };
  H5Dset_extent(pd_dataset, new_size);

  // Write wavefield data for this timestep
  std::vector<float> scalar_data(this->numPoints);
  for (int i = 0; i < this->numPoints; i++) {
    scalar_data[i] = scalars->GetValue(i);
  }

  // Calculate offset and count for this timestep
  hsize_t offset = this->current_timestep * this->numPoints;
  hsize_t count = this->numPoints;

  // Select hyperslab in the file dataset
  hid_t filespace = H5Dget_space(pd_dataset);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);

  // Create memory dataspace
  hid_t memspace = H5Screate_simple(1, &count, NULL);

  // Write data to dataset
  H5Dwrite(pd_dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT,
           scalar_data.data());

  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(pd_dataset);
  H5Gclose(pd_group);

  // Update temporal metadata arrays - extend all arrays
  hsize_t new_timestep_count[1] = { (hsize_t)(this->current_timestep + 1) };

  // Extend and update NumberOfPoints array
  hid_t np_dataset = H5Dopen(vtkhdf_group, "NumberOfPoints", H5P_DEFAULT);
  H5Dset_extent(np_dataset, new_timestep_count);
  hsize_t ts_offset = this->current_timestep;
  hsize_t ts_count = 1;
  hid_t np_filespace = H5Dget_space(np_dataset);
  H5Sselect_hyperslab(np_filespace, H5S_SELECT_SET, &ts_offset, NULL, &ts_count,
                      NULL);
  hid_t np_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long numPoints = this->numPoints;
  H5Dwrite(np_dataset, H5T_NATIVE_LLONG, np_memspace, np_filespace, H5P_DEFAULT,
           &numPoints);
  H5Sclose(np_memspace);
  H5Sclose(np_filespace);
  H5Dclose(np_dataset);

  // Extend and update NumberOfCells array
  hid_t nc_dataset = H5Dopen(vtkhdf_group, "NumberOfCells", H5P_DEFAULT);
  H5Dset_extent(nc_dataset, new_timestep_count);
  hid_t nc_filespace = H5Dget_space(nc_dataset);
  H5Sselect_hyperslab(nc_filespace, H5S_SELECT_SET, &ts_offset, NULL, &ts_count,
                      NULL);
  hid_t nc_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long numCells = this->numCells;
  H5Dwrite(nc_dataset, H5T_NATIVE_LLONG, nc_memspace, nc_filespace, H5P_DEFAULT,
           &numCells);
  H5Sclose(nc_memspace);
  H5Sclose(nc_filespace);
  H5Dclose(nc_dataset);

  // Extend and update NumberOfConnectivityIds array
  hid_t nci_dataset =
      H5Dopen(vtkhdf_group, "NumberOfConnectivityIds", H5P_DEFAULT);
  H5Dset_extent(nci_dataset, new_timestep_count);
  hid_t nci_filespace = H5Dget_space(nci_dataset);
  H5Sselect_hyperslab(nci_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t nci_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long numConnIds = this->numConnectivityIds;
  H5Dwrite(nci_dataset, H5T_NATIVE_LLONG, nci_memspace, nci_filespace,
           H5P_DEFAULT, &numConnIds);
  H5Sclose(nci_memspace);
  H5Sclose(nci_filespace);
  H5Dclose(nci_dataset);

  // Update Steps metadata
  hid_t steps_group = H5Gopen(vtkhdf_group, "Steps", H5P_DEFAULT);

  // Extend and write time value for this timestep
  double time_value = static_cast<double>(istep) * this->dt;
  hid_t values_dataset = H5Dopen(steps_group, "Values", H5P_DEFAULT);
  H5Dset_extent(values_dataset, new_timestep_count);
  hid_t values_filespace = H5Dget_space(values_dataset);
  H5Sselect_hyperslab(values_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t values_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(values_dataset, H5T_NATIVE_DOUBLE, values_memspace, values_filespace,
           H5P_DEFAULT, &time_value);
  H5Sclose(values_memspace);
  H5Sclose(values_filespace);
  H5Dclose(values_dataset);

  // Extend and update NumberOfParts (always 1)
  hid_t nparts_dataset = H5Dopen(steps_group, "NumberOfParts", H5P_DEFAULT);
  H5Dset_extent(nparts_dataset, new_timestep_count);
  hid_t nparts_filespace = H5Dget_space(nparts_dataset);
  H5Sselect_hyperslab(nparts_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t nparts_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long numParts = 1;
  H5Dwrite(nparts_dataset, H5T_NATIVE_LLONG, nparts_memspace, nparts_filespace,
           H5P_DEFAULT, &numParts);
  H5Sclose(nparts_memspace);
  H5Sclose(nparts_filespace);
  H5Dclose(nparts_dataset);

  // Extend and update offset arrays (all zeros for static geometry/single part)
  long long zeroOffset = 0;

  // PartOffsets
  hid_t part_offsets_dataset = H5Dopen(steps_group, "PartOffsets", H5P_DEFAULT);
  H5Dset_extent(part_offsets_dataset, new_timestep_count);
  hid_t part_offsets_filespace = H5Dget_space(part_offsets_dataset);
  H5Sselect_hyperslab(part_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t part_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(part_offsets_dataset, H5T_NATIVE_LLONG, part_offsets_memspace,
           part_offsets_filespace, H5P_DEFAULT, &zeroOffset);
  H5Sclose(part_offsets_memspace);
  H5Sclose(part_offsets_filespace);
  H5Dclose(part_offsets_dataset);

  // PointOffsets
  hid_t point_offsets_dataset =
      H5Dopen(steps_group, "PointOffsets", H5P_DEFAULT);
  H5Dset_extent(point_offsets_dataset, new_timestep_count);
  hid_t point_offsets_filespace = H5Dget_space(point_offsets_dataset);
  H5Sselect_hyperslab(point_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t point_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(point_offsets_dataset, H5T_NATIVE_LLONG, point_offsets_memspace,
           point_offsets_filespace, H5P_DEFAULT, &zeroOffset);
  H5Sclose(point_offsets_memspace);
  H5Sclose(point_offsets_filespace);
  H5Dclose(point_offsets_dataset);

  // CellOffsets
  hid_t cell_offsets_dataset = H5Dopen(steps_group, "CellOffsets", H5P_DEFAULT);
  H5Dset_extent(cell_offsets_dataset, new_timestep_count);
  hid_t cell_offsets_filespace = H5Dget_space(cell_offsets_dataset);
  H5Sselect_hyperslab(cell_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t cell_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(cell_offsets_dataset, H5T_NATIVE_LLONG, cell_offsets_memspace,
           cell_offsets_filespace, H5P_DEFAULT, &zeroOffset);
  H5Sclose(cell_offsets_memspace);
  H5Sclose(cell_offsets_filespace);
  H5Dclose(cell_offsets_dataset);

  // ConnectivityIdOffsets
  hid_t conn_offsets_dataset =
      H5Dopen(steps_group, "ConnectivityIdOffsets", H5P_DEFAULT);
  H5Dset_extent(conn_offsets_dataset, new_timestep_count);
  hid_t conn_offsets_filespace = H5Dget_space(conn_offsets_dataset);
  H5Sselect_hyperslab(conn_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t conn_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(conn_offsets_dataset, H5T_NATIVE_LLONG, conn_offsets_memspace,
           conn_offsets_filespace, H5P_DEFAULT, &zeroOffset);
  H5Sclose(conn_offsets_memspace);
  H5Sclose(conn_offsets_filespace);
  H5Dclose(conn_offsets_dataset);

  // Update PointDataOffsets/Wavefield
  hid_t pd_offsets_group =
      H5Gopen(steps_group, "PointDataOffsets", H5P_DEFAULT);
  hid_t wf_offsets_dataset =
      H5Dopen(pd_offsets_group, "Wavefield", H5P_DEFAULT);
  H5Dset_extent(wf_offsets_dataset, new_timestep_count);
  hid_t wf_offsets_filespace = H5Dget_space(wf_offsets_dataset);
  H5Sselect_hyperslab(wf_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t wf_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long wavefieldOffset = this->current_timestep * this->numPoints;
  H5Dwrite(wf_offsets_dataset, H5T_NATIVE_LLONG, wf_offsets_memspace,
           wf_offsets_filespace, H5P_DEFAULT, &wavefieldOffset);
  H5Sclose(wf_offsets_memspace);
  H5Sclose(wf_offsets_filespace);
  H5Dclose(wf_offsets_dataset);
  H5Gclose(pd_offsets_group);

  // Update NSteps attribute
  int nsteps_written = this->current_timestep + 1;
  hid_t attr = H5Aopen(steps_group, "NSteps", H5P_DEFAULT);
  H5Awrite(attr, H5T_NATIVE_INT, &nsteps_written);
  H5Aclose(attr);

  H5Gclose(steps_group);

  // Close HDF5 resources
  H5Gclose(vtkhdf_group);
  H5Fclose(hdf5_file_id);

  this->current_timestep++;

  specfem::Logger::info("Wrote 3D wavefield data for timestep " +
                        std::to_string(istep) + " to HDF5 file (step " +
                        std::to_string(this->current_timestep) + ")");

#else
  throw std::runtime_error(
      "VTK HDF5 output requested but HDF5 support not compiled.");
#endif
}

void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    run(specfem::assembly::assembly<dimension_tag> &assembly, const int istep) {

  // Update the wavefield scalars only
  auto scalars = compute_wavefield_scalars(assembly);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  switch (output_format) {
  case (display::format::vtkhdf):
    this->run<specfem::display::format::vtkhdf>(scalars, istep);
    break;

  case (display::format::on_screen):
    this->run<specfem::display::format::on_screen>(scalars, istep);
    break;

  case (display::format::PNG):
    this->run<specfem::display::format::PNG>(scalars, istep);
    break;

  case (display::format::JPG):
    this->run<specfem::display::format::JPG>(scalars, istep);
    break;

  default:
    throw std::runtime_error("Unsupported output format for 3D");
  }
}

void specfem::periodic_tasks::plot_wavefield<specfem::dimension::type::dim3>::
    finalize(specfem::assembly::assembly<dimension_tag> &assembly) {

  // Clean up VTK objects
  unstructured_grid = nullptr;
}

#endif // NO_VTK
