#include "plot_wavefield.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
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

namespace specfem::periodic_tasks::plot_wavefield_impl {
/// @brief Check HDF5 return codes and abort on failure
inline void h5_check(herr_t err, const char *call, int line, const char *file) {
  if (err < 0) {
    std::ostringstream oss;
    oss << "HDF5 error in " << call << " at " << file << ":" << line;
    specfem::program::abort(oss.str(), 30, line, file);
  }
}

/// @brief Check HDF5 identifier return values and abort on failure
inline hid_t h5_check_id(hid_t id, const char *call, int line,
                         const char *file) {
  if (id < 0) {
    std::ostringstream oss;
    oss << "HDF5 error in " << call << " at " << file << ":" << line;
    specfem::program::abort(oss.str(), 30, line, file);
  }
  return id;
}
} // namespace specfem::periodic_tasks::plot_wavefield_impl

#define SPECFEM_H5_CHECK(call)                                                 \
  specfem::periodic_tasks::plot_wavefield_impl::h5_check((call), #call,        \
                                                         __LINE__, __FILE__)
#define SPECFEM_H5_CHECK_ID(call)                                              \
  specfem::periodic_tasks::plot_wavefield_impl::h5_check_id(                   \
      (call), #call, __LINE__, __FILE__)

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

// Get wavefield type to display
specfem::enums::wavefield specfem::periodic_tasks::plot_wavefield<
    specfem::element::dimension_tag::dim3>::get_wavefield_type() {
  if (wavefield_type == specfem::enums::wavefield::displacement) {
    return specfem::enums::wavefield::displacement;
  } else if (wavefield_type == specfem::enums::wavefield::velocity) {
    return specfem::enums::wavefield::velocity;
  } else if (wavefield_type == specfem::enums::wavefield::acceleration) {
    return specfem::enums::wavefield::acceleration;
  } else {
    throw std::runtime_error("Wavefield type not supported");
  }
}

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
void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::initialize<
        specfem::enums::display_format::vtkhdf>(
        vtkSmartPointer<vtkFloatArray> &scalars) {

#ifndef NO_HDF5

  // Initialize VTK HDF5 file for time series output
  this->current_timestep = 0;
  this->numPoints = 0;
  this->numCells = 0;

  // Create HDF5 file
  this->hdf5_filename = (this->output_folder / "wavefield.vtkhdf").string();
  hid_t hdf5_file_id = SPECFEM_H5_CHECK_ID(H5Fcreate(
      this->hdf5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
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
  hid_t dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(2, point_dims, NULL));
  hid_t dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(vtkhdf_group, "Points", H5T_NATIVE_DOUBLE, dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, pointCoords.data()));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(dataspace));

  // Write Connectivity (static)
  hsize_t dims[1];
  dims[0] = connectivity.size();
  dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(vtkhdf_group, "Connectivity", H5T_NATIVE_LLONG, dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, connectivity.data()));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(dataspace));

  // Write Offsets (static)
  dims[0] = offsets.size();
  dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(vtkhdf_group, "Offsets", H5T_NATIVE_LLONG, dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, offsets.data()));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(dataspace));

  // Write Types (static)
  dims[0] = types.size();
  dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(vtkhdf_group, "Types", H5T_NATIVE_UCHAR, dataspace, H5P_DEFAULT,
                H5P_DEFAULT, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dwrite(dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, types.data()));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(dataspace));

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
  dims[0] = material_ids.size();
  dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(cd_group, "MaterialID", H5T_NATIVE_INT, dataspace, H5P_DEFAULT,
                H5P_DEFAULT, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, material_ids.data()));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(dataspace));
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

    dims[0] = jacobian_data.size();
    dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
    dataset = SPECFEM_H5_CHECK_ID(
        H5Dcreate(pd_group, "Jacobian", H5T_NATIVE_FLOAT, dataspace,
                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    SPECFEM_H5_CHECK(H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, jacobian_data.data()));
    SPECFEM_H5_CHECK(H5Dclose(dataset));
    SPECFEM_H5_CHECK(H5Sclose(dataspace));
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
    dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
    dataset = SPECFEM_H5_CHECK_ID(H5Dcreate(pd_group, "Kappa", H5T_NATIVE_FLOAT,
                                            dataspace, H5P_DEFAULT, H5P_DEFAULT,
                                            H5P_DEFAULT));
    SPECFEM_H5_CHECK(H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, kappa_data.data()));
    SPECFEM_H5_CHECK(H5Dclose(dataset));
    SPECFEM_H5_CHECK(H5Sclose(dataspace));

    // Write mu
    dims[0] = mu_data.size();
    dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
    dataset = SPECFEM_H5_CHECK_ID(H5Dcreate(pd_group, "Mu", H5T_NATIVE_FLOAT,
                                            dataspace, H5P_DEFAULT, H5P_DEFAULT,
                                            H5P_DEFAULT));
    SPECFEM_H5_CHECK(H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, mu_data.data()));
    SPECFEM_H5_CHECK(H5Dclose(dataset));
    SPECFEM_H5_CHECK(H5Sclose(dataspace));

    // Write rho
    dims[0] = rho_data.size();
    dataspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, dims, NULL));
    dataset = SPECFEM_H5_CHECK_ID(H5Dcreate(pd_group, "Rho", H5T_NATIVE_FLOAT,
                                            dataspace, H5P_DEFAULT, H5P_DEFAULT,
                                            H5P_DEFAULT));
    SPECFEM_H5_CHECK(H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, rho_data.data()));
    SPECFEM_H5_CHECK(H5Dclose(dataset));
    SPECFEM_H5_CHECK(H5Sclose(dataspace));
  }

  // Create extensible dataset for wavefield scalars
  // Initial size: 0 (will grow as needed)
  hsize_t pd_initial_dims[1] = { 0 };
  hsize_t pd_max_dims[1] = { H5S_UNLIMITED };
  hid_t pd_dataspace =
      SPECFEM_H5_CHECK_ID(H5Screate_simple(1, pd_initial_dims, pd_max_dims));

  // Create dataset creation property list and set chunking
  hid_t pd_plist = SPECFEM_H5_CHECK_ID(H5Pcreate(H5P_DATASET_CREATE));
  hsize_t pd_chunk_dims[1] = {
    (hsize_t)this->numPoints
  }; // Chunk size = one timestep worth of data
  SPECFEM_H5_CHECK(H5Pset_chunk(pd_plist, 1, pd_chunk_dims));

  hid_t pd_dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(pd_group, "Wavefield", H5T_NATIVE_FLOAT, pd_dataspace,
                H5P_DEFAULT, pd_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(pd_dataset));
  SPECFEM_H5_CHECK(H5Pclose(pd_plist));
  SPECFEM_H5_CHECK(H5Sclose(pd_dataspace));
  SPECFEM_H5_CHECK(H5Gclose(pd_group));

  // Create extensible temporal metadata arrays instead of pre-allocated ones
  hsize_t temp_initial_dims[1] = { 0 };
  hsize_t temp_max_dims[1] = { H5S_UNLIMITED };
  hsize_t temp_chunk_dims[1] = { 1 }; // Chunk size = 1 timestep

  // Create dataset creation property list for chunking
  hid_t temp_plist = SPECFEM_H5_CHECK_ID(H5Pcreate(H5P_DATASET_CREATE));
  SPECFEM_H5_CHECK(H5Pset_chunk(temp_plist, 1, temp_chunk_dims));

  hid_t temp_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, temp_initial_dims, temp_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
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

  // Create extensible datasets for NumberOfParts and offset arrays
  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "NumberOfParts", H5T_NATIVE_LLONG, steps_dataspace,
                H5P_DEFAULT, steps_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "PartOffsets", H5T_NATIVE_LLONG, steps_dataspace,
                H5P_DEFAULT, steps_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "PointOffsets", H5T_NATIVE_LLONG, steps_dataspace,
                H5P_DEFAULT, steps_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "CellOffsets", H5T_NATIVE_LLONG, steps_dataspace,
                H5P_DEFAULT, steps_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(steps_group, "ConnectivityIdOffsets", H5T_NATIVE_LLONG,
                steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));

  // Create PointDataOffsets subgroup with extensible datasets
  hid_t pd_offsets_group = SPECFEM_H5_CHECK_ID(H5Gcreate(
      steps_group, "PointDataOffsets", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  // Wavefield offsets (extensible)
  steps_dataspace = SPECFEM_H5_CHECK_ID(
      H5Screate_simple(1, steps_initial_dims, steps_max_dims));
  dataset = SPECFEM_H5_CHECK_ID(
      H5Dcreate(pd_offsets_group, "Wavefield", H5T_NATIVE_LLONG,
                steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dclose(dataset));
  SPECFEM_H5_CHECK(H5Sclose(steps_dataspace));
  SPECFEM_H5_CHECK(H5Gclose(pd_offsets_group));

  SPECFEM_H5_CHECK(H5Pclose(steps_plist));
  SPECFEM_H5_CHECK(H5Gclose(steps_group));

  // Close HDF5 file - will reopen for each timestep write
  SPECFEM_H5_CHECK(H5Gclose(vtkhdf_group));
  SPECFEM_H5_CHECK(H5Fclose(hdf5_file_id));

  specfem::Logger::info([&](std::ostringstream &oss) {
    oss << "Initialized VTK HDF5 file for 3D wavefield output: "
        << this->hdf5_filename << " (extensible datasets)";
  });

#else
  throw std::runtime_error(
      "VTK HDF5 output requested but HDF5 support not compiled.");
#endif
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::initialize_display(
        vtkSmartPointer<vtkFloatArray> &scalars) {
  // Not implemented for 3D - only HDF5 output is supported
  throw std::runtime_error(
      "Display initialization not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::initialize<
        specfem::enums::display_format::on_screen>(
        vtkSmartPointer<vtkFloatArray> &scalars) {
  throw std::runtime_error(
      "On-screen display not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::initialize<
        specfem::enums::display_format::PNG>(
        vtkSmartPointer<vtkFloatArray> &scalars) {
  throw std::runtime_error(
      "PNG output not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::initialize<
        specfem::enums::display_format::JPG>(
        vtkSmartPointer<vtkFloatArray> &scalars) {
  throw std::runtime_error(
      "JPG output not supported for 3D. Use VTK HDF5 output.");
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
    this->initialize<specfem::enums::display_format::vtkhdf>(scalars);
    break;
  case specfem::enums::display_format::on_screen:
    this->initialize<specfem::enums::display_format::on_screen>(scalars);
    break;
  case specfem::enums::display_format::PNG:
    this->initialize<specfem::enums::display_format::PNG>(scalars);
    break;
  case specfem::enums::display_format::JPG:
    this->initialize<specfem::enums::display_format::JPG>(scalars);
    break;
  default:
    throw std::runtime_error("Unsupported display format for 3D");
  }

  return;
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::run_render(
        vtkSmartPointer<vtkFloatArray> &scalars) {
  // Not implemented for 3D
  throw std::runtime_error(
      "Rendering not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::run<
        specfem::enums::display_format::on_screen>(
        vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {
  throw std::runtime_error(
      "On-screen display not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::run<
        specfem::enums::display_format::PNG>(
        vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {
  throw std::runtime_error(
      "PNG output not supported for 3D. Use VTK HDF5 output.");
}

template <>
void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::run<
        specfem::enums::display_format::JPG>(
        vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {
  throw std::runtime_error(
      "JPG output not supported for 3D. Use VTK HDF5 output.");
}

// Helper: extend a 1D dataset, select hyperslab at write_offset, write one
// scalar value, then close the dataset.
void specfem::periodic_tasks::plot_wavefield<
    specfem::element::dimension_tag::dim3>::
    extend_and_write_scalar(hid_t parent, const char *dataset_name,
                            hsize_t new_extent, hsize_t write_offset,
                            hid_t mem_type, const void *data) {
#ifndef NO_HDF5
  hid_t ds = SPECFEM_H5_CHECK_ID(H5Dopen(parent, dataset_name, H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Dset_extent(ds, &new_extent));

  hid_t filespace = SPECFEM_H5_CHECK_ID(H5Dget_space(ds));
  hsize_t count = 1;
  SPECFEM_H5_CHECK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &write_offset,
                                       NULL, &count, NULL));
  hid_t memspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, &count, NULL));
  SPECFEM_H5_CHECK(
      H5Dwrite(ds, mem_type, memspace, filespace, H5P_DEFAULT, data));

  SPECFEM_H5_CHECK(H5Sclose(memspace));
  SPECFEM_H5_CHECK(H5Sclose(filespace));
  SPECFEM_H5_CHECK(H5Dclose(ds));
#endif
}

template <>
void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::run<
        specfem::enums::display_format::vtkhdf>(
        vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {

#ifndef NO_HDF5
  // Open HDF5 file for extending datasets
  hid_t hdf5_file_id = SPECFEM_H5_CHECK_ID(
      H5Fopen(this->hdf5_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
  hid_t vtkhdf_group =
      SPECFEM_H5_CHECK_ID(H5Gopen(hdf5_file_id, "/VTKHDF", H5P_DEFAULT));

  // Extend and write wavefield data
  hid_t pd_group =
      SPECFEM_H5_CHECK_ID(H5Gopen(vtkhdf_group, "PointData", H5P_DEFAULT));
  hid_t pd_dataset =
      SPECFEM_H5_CHECK_ID(H5Dopen(pd_group, "Wavefield", H5P_DEFAULT));

  // Extend the wavefield dataset to accommodate new timestep
  hsize_t new_size[1] = { (hsize_t)((this->current_timestep + 1) *
                                    this->numPoints) };
  SPECFEM_H5_CHECK(H5Dset_extent(pd_dataset, new_size));

  // Write wavefield data for this timestep
  std::vector<float> scalar_data(this->numPoints);
  for (long long i = 0; i < this->numPoints; i++) {
    scalar_data[i] = scalars->GetValue(i);
  }

  // Calculate offset and count for this timestep
  hsize_t offset = this->current_timestep * this->numPoints;
  hsize_t count = this->numPoints;

  // Select hyperslab in the file dataset
  hid_t filespace = SPECFEM_H5_CHECK_ID(H5Dget_space(pd_dataset));
  SPECFEM_H5_CHECK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL,
                                       &count, NULL));

  // Create memory dataspace and write
  hid_t memspace = SPECFEM_H5_CHECK_ID(H5Screate_simple(1, &count, NULL));
  SPECFEM_H5_CHECK(H5Dwrite(pd_dataset, H5T_NATIVE_FLOAT, memspace, filespace,
                            H5P_DEFAULT, scalar_data.data()));

  SPECFEM_H5_CHECK(H5Sclose(memspace));
  SPECFEM_H5_CHECK(H5Sclose(filespace));
  SPECFEM_H5_CHECK(H5Dclose(pd_dataset));
  SPECFEM_H5_CHECK(H5Gclose(pd_group));

  // Update temporal metadata arrays
  hsize_t new_ts_count = (hsize_t)(this->current_timestep + 1);
  hsize_t ts_offset = this->current_timestep;

  long long num_points_val = this->numPoints;
  long long num_cells_val = this->numCells;
  long long num_conn_ids_val = this->numConnectivityIds;

  extend_and_write_scalar(vtkhdf_group, "NumberOfPoints", new_ts_count,
                          ts_offset, H5T_NATIVE_LLONG, &num_points_val);
  extend_and_write_scalar(vtkhdf_group, "NumberOfCells", new_ts_count,
                          ts_offset, H5T_NATIVE_LLONG, &num_cells_val);
  extend_and_write_scalar(vtkhdf_group, "NumberOfConnectivityIds", new_ts_count,
                          ts_offset, H5T_NATIVE_LLONG, &num_conn_ids_val);

  // Update Steps metadata
  hid_t steps_group =
      SPECFEM_H5_CHECK_ID(H5Gopen(vtkhdf_group, "Steps", H5P_DEFAULT));

  double time_value =
      static_cast<double>(this->assembly.t0) +
      static_cast<double>(istep) * static_cast<double>(this->assembly.dt);
  extend_and_write_scalar(steps_group, "Values", new_ts_count, ts_offset,
                          H5T_NATIVE_DOUBLE, &time_value);

  long long num_parts = 1;
  extend_and_write_scalar(steps_group, "NumberOfParts", new_ts_count, ts_offset,
                          H5T_NATIVE_LLONG, &num_parts);

  // Offset arrays (all zeros for static geometry / single part)
  long long zero_offset = 0;
  extend_and_write_scalar(steps_group, "PartOffsets", new_ts_count, ts_offset,
                          H5T_NATIVE_LLONG, &zero_offset);
  extend_and_write_scalar(steps_group, "PointOffsets", new_ts_count, ts_offset,
                          H5T_NATIVE_LLONG, &zero_offset);
  extend_and_write_scalar(steps_group, "CellOffsets", new_ts_count, ts_offset,
                          H5T_NATIVE_LLONG, &zero_offset);
  extend_and_write_scalar(steps_group, "ConnectivityIdOffsets", new_ts_count,
                          ts_offset, H5T_NATIVE_LLONG, &zero_offset);

  // Update PointDataOffsets/Wavefield
  hid_t pd_offsets_group = SPECFEM_H5_CHECK_ID(
      H5Gopen(steps_group, "PointDataOffsets", H5P_DEFAULT));
  long long wavefield_offset = this->current_timestep * this->numPoints;
  extend_and_write_scalar(pd_offsets_group, "Wavefield", new_ts_count,
                          ts_offset, H5T_NATIVE_LLONG, &wavefield_offset);
  SPECFEM_H5_CHECK(H5Gclose(pd_offsets_group));

  // Update NSteps attribute
  int nsteps_written = this->current_timestep + 1;
  hid_t attr = SPECFEM_H5_CHECK_ID(H5Aopen(steps_group, "NSteps", H5P_DEFAULT));
  SPECFEM_H5_CHECK(H5Awrite(attr, H5T_NATIVE_INT, &nsteps_written));
  SPECFEM_H5_CHECK(H5Aclose(attr));

  SPECFEM_H5_CHECK(H5Gclose(steps_group));

  // Close HDF5 resources
  SPECFEM_H5_CHECK(H5Gclose(vtkhdf_group));
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
  case (specfem::enums::display_format::vtkhdf):
    this->run<specfem::enums::display_format::vtkhdf>(scalars, istep);
    break;

  case (specfem::enums::display_format::on_screen):
    this->run<specfem::enums::display_format::on_screen>(scalars, istep);
    break;

  case (specfem::enums::display_format::PNG):
    this->run<specfem::enums::display_format::PNG>(scalars, istep);
    break;

  case (specfem::enums::display_format::JPG):
    this->run<specfem::enums::display_format::JPG>(scalars, istep);
    break;

  default:
    throw std::runtime_error("Unsupported output format for 3D");
  }
}

void specfem::periodic_tasks::
    plot_wavefield<specfem::element::dimension_tag::dim3>::finalize(
        specfem::assembly::assembly<dimension_tag> &assembly) {

  // Clean up VTK objects
  unstructured_grid = nullptr;
}

#endif // NO_VTK
