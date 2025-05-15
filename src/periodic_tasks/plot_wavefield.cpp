#include "periodic_tasks/plot_wavefield.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"
#include "periodic_tasks/plotter.hpp"
#include "utilities/strings.hpp"

#ifdef NO_VTK

#include <sstream>

#else

#include <algorithm>
#include <boost/filesystem.hpp>
#include <cmath>
#include <vtkActor.h>
#include <vtkBiQuadraticQuad.h>
#include <vtkCellData.h>
#include <vtkDataSetMapper.h>
#include <vtkExtractEdges.h>
#include <vtkFloatArray.h>
#include <vtkGraphicsFactory.h>
#include <vtkJPEGWriter.h>
#include <vtkLookupTable.h>
#include <vtkNamedColors.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkQuad.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWindowToImageFilter.h>

#endif // NO_VTK

#ifdef NO_VTK

// Add this constructor implementation for NO_VTK builds
specfem::periodic_tasks::plot_wavefield::plot_wavefield(
    const specfem::compute::assembly &assembly,
    const specfem::display::format &output_format,
    const specfem::display::wavefield &component,
    const specfem::wavefield::simulation_field &wavefield,
    const int &time_interval, const boost::filesystem::path &output_folder,
    specfem::MPI::MPI *mpi)
    : assembly(assembly), wavefield(wavefield), component(component),
      plotter(time_interval), output_format(output_format),
      output_folder(output_folder), nspec(assembly.mesh.nspec),
      ngllx(assembly.mesh.ngllx), ngllz(assembly.mesh.ngllz), mpi(mpi) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::plot_wavefield::run(
    specfem::compute::assembly &assembly, const int istep) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::plot_wavefield::initialize(
    specfem::compute::assembly &assembly) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::plot_wavefield::finalize(
    specfem::compute::assembly &assembly) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

#else

// Constructor
specfem::periodic_tasks::plot_wavefield::plot_wavefield(
    const specfem::compute::assembly &assembly,
    const specfem::display::format &output_format,
    const specfem::display::wavefield &component,
    const specfem::wavefield::simulation_field &wavefield,
    const int &time_interval, const boost::filesystem::path &output_folder,
    specfem::MPI::MPI *mpi)
    : assembly(assembly), wavefield(wavefield), component(component),
      plotter(time_interval), output_format(output_format),
      output_folder(output_folder), nspec(assembly.mesh.nspec),
      ngllx(assembly.mesh.ngllx), ngllz(assembly.mesh.ngllz), mpi(mpi) {};

// Sigmoid function centered at 0.0
double specfem::periodic_tasks::plot_wavefield::sigmoid(double x) {
  return (1 / (1 + std::exp(-100 * x)) - 0.5) * 1.5;
}

// Get wavefield component type from display component
specfem::wavefield::type
specfem::periodic_tasks::plot_wavefield::get_wavefield_component() {
  if (component == specfem::display::wavefield::displacement) {
    return specfem::wavefield::type::displacement;
  } else if (component == specfem::display::wavefield::velocity) {
    return specfem::wavefield::type::velocity;
  } else if (component == specfem::display::wavefield::acceleration) {
    return specfem::wavefield::type::acceleration;
  } else if (component == specfem::display::wavefield::pressure) {
    return specfem::wavefield::type::pressure;
  } else {
    throw std::runtime_error("Unsupported component");
  }
}

// Maps different materials to different colors
vtkSmartPointer<vtkDataSetMapper>
specfem::periodic_tasks::plot_wavefield::map_materials_with_color() {

  const auto &element_types = assembly.element_types;

  const std::unordered_map<specfem::element::medium_tag, std::array<int, 3> >
      material_colors = {
        { specfem::element::medium_tag::acoustic, // aqua color
          { 0, 255, 255 } },
        { specfem::element::medium_tag::elastic_psv, // sienna color
          { 160, 82, 45 } },
        { specfem::element::medium_tag::elastic_sh, // sienna color
          { 160, 82, 45 } },
        { specfem::element::medium_tag::poroelastic, // off navy color
          { 40, 40, 128 } },
        { specfem::element::medium_tag::electromagnetic_te, // dark gray color
          { 169, 169, 169 } },
      };

  const auto &coordinates = assembly.mesh.points.h_coord;
  const int nspec = assembly.mesh.nspec;
  const int ngllx = assembly.mesh.ngllx;
  const int ngllz = assembly.mesh.ngllz;

  const int cell_points = 4;

  const std::array<int, cell_points> z_index = { 0, ngllz - 1, ngllz - 1, 0 };
  const std::array<int, cell_points> x_index = { 0, 0, ngllx - 1, ngllx - 1 };

  auto points = vtkSmartPointer<vtkPoints>::New();

  auto cells = vtkSmartPointer<vtkCellArray>::New();

  auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
  colors->SetNumberOfComponents(3);
  colors->SetName("Colors");

  for (int icell = 0; icell < nspec; ++icell) {
    for (int i = 0; i < cell_points; ++i) {
      points->InsertNextPoint(coordinates(0, icell, z_index[i], x_index[i]),
                              coordinates(1, icell, z_index[i], x_index[i]),
                              0.0);
    }
    auto quad = vtkSmartPointer<vtkQuad>::New();
    for (int i = 0; i < cell_points; ++i) {
      quad->GetPointIds()->SetId(i, icell * cell_points + i);
    }
    cells->InsertNextCell(quad);

    const auto material = element_types.get_medium_tag(icell);
    const auto color = material_colors.at(material);
    unsigned char color_uc[3] = { static_cast<unsigned char>(color[0]),
                                  static_cast<unsigned char>(color[1]),
                                  static_cast<unsigned char>(color[2]) };
    colors->InsertNextTypedTuple(color_uc);
  }

  auto unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  unstructured_grid->SetPoints(points);
  unstructured_grid->SetCells(VTK_QUAD, cells);

  unstructured_grid->GetCellData()->SetScalars(colors);

  auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
  mapper->SetInputData(unstructured_grid);

  return mapper;
}

/**
 * @brief Get the wavefield on vtkUnstructured grid object as biquadratic quads

 * This function creates bilinear quadrilateral from the element corners,
 * midpoints and center points of element sides.
 *
 * Graphical Explanation looking at a single element (see below), create:
 *
 *
 *     3----•-----6-----•----2
 *     |    |     |     |    |
 *     •----•-----•-----•----•
 *     |    |     |     |    |
 *     7----•-----8-----•----5
 *     |    |     |     |    |
 *     •----•-----•-----•----•
 *     |    |     |     |    |
 *     0----•-----4-----•----1
 *
 * Where the above points (for GLL = 5) that are used to create the bilinear
 * quad are indicated by numbers 0-8 in the order of the points in the quad.
 * Each element has therefore 9 points, that are the used to return a
 * vtkUnstructuredGrid object containing vtkBiQuadraticQuad cells.
 *
 * @param assembly
 * @param type
 * @param display_component
 * @return vtkSmartPointer<vtkUnstructuredGrid>
 */
void specfem::periodic_tasks::plot_wavefield::create_biquad_grid() {
  const auto &coordinates = assembly.mesh.points.h_coord;

  const int ncells = nspec;
  const int cell_points = 9;

  const std::array<int, cell_points> z_index = { 0,
                                                 0,
                                                 ngllz - 1,
                                                 ngllz - 1,
                                                 0,
                                                 (ngllz - 1) / 2,
                                                 ngllz - 1,
                                                 (ngllz - 1) / 2,
                                                 (ngllz - 1) / 2 };
  const std::array<int, cell_points> x_index = { 0,
                                                 ngllx - 1,
                                                 ngllx - 1,
                                                 0,
                                                 (ngllx - 1) / 2,
                                                 ngllx - 1,
                                                 (ngllx - 1) / 2,
                                                 0,
                                                 (ngllx - 1) / 2 };

  auto points = vtkSmartPointer<vtkPoints>::New();
  auto cells = vtkSmartPointer<vtkCellArray>::New();

  for (int icell = 0; icell < ncells; ++icell) {
    for (int i = 0; i < cell_points; ++i) {
      points->InsertNextPoint(coordinates(0, icell, z_index[i], x_index[i]),
                              coordinates(1, icell, z_index[i], x_index[i]),
                              0.0);
    }
    auto quad = vtkSmartPointer<vtkBiQuadraticQuad>::New();
    for (int i = 0; i < cell_points; ++i) {
      quad->GetPointIds()->SetId(i, icell * cell_points + i);
    }
    cells->InsertNextCell(quad);
  }

  unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  unstructured_grid->SetPoints(points);
  unstructured_grid->SetCells(VTK_BIQUADRATIC_QUAD, cells);
}

/**
 * @brief Get the wavefield on vtkUnstructured grid object
 *
 *
 * This function creates vertices for quadrilaterals from the coordinates x and
 * z. and the element based field. The field is of shape (nspec, ngll, ngll), so
 * are the coordinates. The functions creates quads of 4 GLL points. For ngll =
 * 5 this means that we have 16 quads per element, and a total of nspec * 16
 * quads.
 *
 * Graphical Explanation:
 *
 * Looking at a single element (see below), create quadrilateral for each
 * subrectangle of the element. Starting with the ix=0, iz=0 corner moving
 * counterclockwise for each subquad, indicated by the numbers coinciding with
 * the GLL points. Then we move in ix direction for each quad indicated by the
 * number on the face of each quad.
 *
 *     •----•-----•-----3----2
 *     | 12 |  13 |  14 | 15 |
 *     •----•-----•-----0----1
 *     |  8 |   9 |  10 | 11 |
 *     •----•-----•-----•----•
 *     |  4 |   5 |   6 |  7 |
 *     3----2-----•-----•----•
 *     |  0 |   1 |   2 |  3 |
 *     0----1-----•-----•----•
 *
 * So, for GLL = 5 each element each element has therefore 16 (as numbered 0-15)
 * quads. For the first and last quad we indicate the order of the gll points
 * used as vertices of the quad (0-3). Finally, the quads are the used to return
 * a vtkUnstructuredGrid object containing vtkQuad cells.
 *
 * The wavefield is assigned to the points accordingly.
 *
 * @param assembly
 * @param type
 * @param display_component
 * @return vtkSmartPointer<vtkUnstructuredGrid>
 */
void specfem::periodic_tasks::plot_wavefield::create_quad_grid() {
  const auto &coordinates = assembly.mesh.points.h_coord;

  // For ngll = 5, each spectral element has 16 cells
  const int n_cells_per_spec = (ngllx - 1) * (ngllz - 1);
  const int ncells = nspec * n_cells_per_spec;

  const int n_cell_points = 4;

  // The points of the cells are ordered as follows:
  // 3--2
  // |  |
  // 0--1
  const std::array<int, n_cell_points> z_index = { 0, 0, 1, 1 };
  const std::array<int, n_cell_points> x_index = { 0, 1, 1, 0 };

  auto points = vtkSmartPointer<vtkPoints>::New();
  auto cells = vtkSmartPointer<vtkCellArray>::New();

  int point_counter = 0; // Keep track of the global point index

  // Loop over the cells
  for (int ispec = 0; ispec < nspec; ++ispec) {
    for (int iz = 0; iz < ngllz - 1; ++iz) {
      for (int ix = 0; ix < ngllx - 1; ++ix) {
        auto quad = vtkSmartPointer<vtkQuad>::New();

        for (int ipoint = 0; ipoint < n_cell_points; ++ipoint) {
          int iz_pos = iz + z_index[ipoint];
          int ix_pos = ix + x_index[ipoint];

          // Insert the point
          points->InsertNextPoint(coordinates(0, ispec, iz_pos, ix_pos),
                                  coordinates(1, ispec, iz_pos, ix_pos), 0.0);

          // Set the point ID for this quad
          quad->GetPointIds()->SetId(ipoint, point_counter);
          point_counter++;
        }

        // Add the cell
        cells->InsertNextCell(quad);
      }
    }
  }

  // Create the unstructured grid
  unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  unstructured_grid->SetPoints(points);
  unstructured_grid->SetCells(VTK_QUAD, cells);
}

// Compute wavefield scalar values for the grid points
vtkSmartPointer<vtkFloatArray>
specfem::periodic_tasks::plot_wavefield::compute_wavefield_scalars(
    specfem::compute::assembly &assembly) {
  const auto component_type = get_wavefield_component();
  const auto &wavefield_data =
      assembly.generate_wavefield_on_entire_grid(wavefield, component_type);

  auto scalars = vtkSmartPointer<vtkFloatArray>::New();

  // For quad grid
  if (unstructured_grid->GetCellType(0) == VTK_QUAD) {
    const int n_cell_points = 4;
    const std::array<int, n_cell_points> z_index = { 0, 0, 1, 1 };
    const std::array<int, n_cell_points> x_index = { 0, 1, 1, 0 };

    // Loop over the cells
    for (int ispec = 0; ispec < nspec; ++ispec) {
      for (int iz = 0; iz < ngllz - 1; ++iz) {
        for (int ix = 0; ix < ngllx - 1; ++ix) {
          for (int ipoint = 0; ipoint < n_cell_points; ++ipoint) {
            int iz_pos = iz + z_index[ipoint];
            int ix_pos = ix + x_index[ipoint];

            // Insert scalar value
            if (component_type == specfem::wavefield::type::pressure) {
              scalars->InsertNextValue(
                  std::abs(wavefield_data(ispec, iz_pos, ix_pos, 0)));
            } else {
              scalars->InsertNextValue(
                  std::sqrt((wavefield_data(ispec, iz_pos, ix_pos, 0) *
                             wavefield_data(ispec, iz_pos, ix_pos, 0)) +
                            (wavefield_data(ispec, iz_pos, ix_pos, 1) *
                             wavefield_data(ispec, iz_pos, ix_pos, 1))));
            }
          }
        }
      }
    }
  }
  // For biquadratic grid
  else if (unstructured_grid->GetCellType(0) == VTK_BIQUADRATIC_QUAD) {
    const int cell_points = 9;
    const std::array<int, cell_points> z_index = { 0,
                                                   0,
                                                   ngllz - 1,
                                                   ngllz - 1,
                                                   0,
                                                   (ngllz - 1) / 2,
                                                   ngllz - 1,
                                                   (ngllz - 1) / 2,
                                                   (ngllz - 1) / 2 };
    const std::array<int, cell_points> x_index = { 0,
                                                   ngllx - 1,
                                                   ngllx - 1,
                                                   0,
                                                   (ngllx - 1) / 2,
                                                   ngllx - 1,
                                                   (ngllx - 1) / 2,
                                                   0,
                                                   (ngllx - 1) / 2 };

    for (int icell = 0; icell < nspec; ++icell) {
      for (int i = 0; i < cell_points; ++i) {
        if (component_type == specfem::wavefield::type::pressure) {
          scalars->InsertNextValue(
              std::abs(wavefield_data(icell, z_index[i], x_index[i], 0)));
        } else {
          scalars->InsertNextValue(
              std::sqrt((wavefield_data(icell, z_index[i], x_index[i], 0) *
                         wavefield_data(icell, z_index[i], x_index[i], 0)) +
                        (wavefield_data(icell, z_index[i], x_index[i], 1) *
                         wavefield_data(icell, z_index[i], x_index[i], 1))));
        }
      }
    }
  }

  return scalars;
}

void specfem::periodic_tasks::plot_wavefield::initialize(
    specfem::compute::assembly &assembly) {

  // Create VTK objects that will persist between calls
  colors = vtkSmartPointer<vtkNamedColors>::New();

  if (output_format != specfem::display::format::on_screen) {
    vtkSmartPointer<vtkGraphicsFactory> graphics_factory;
    graphics_factory->SetOffScreenOnlyMode(1);
    graphics_factory->SetUseMesaClasses(1);
  }

  // Create material mapper and actor
  material_mapper = map_materials_with_color();
  material_actor = vtkSmartPointer<vtkActor>::New();
  material_actor->SetMapper(material_mapper);

  // Create the grid structure
  create_quad_grid(); // or create_biquad_grid() based on preference

  // Compute initial wavefield scalars and add to grid
  auto scalars = compute_wavefield_scalars(assembly);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  // Create lookup table
  lut = vtkSmartPointer<vtkLookupTable>::New();
  lut->SetNumberOfTableValues(256);
  lut->Build();

  // Create a mapper for the wavefield
  wavefield_mapper = vtkSmartPointer<vtkDataSetMapper>::New();
  wavefield_mapper->SetInputData(unstructured_grid);
  wavefield_mapper->SetLookupTable(lut);
  wavefield_mapper->SetScalarModeToUsePointData();
  wavefield_mapper->SetColorModeToMapScalars();
  wavefield_mapper->SetScalarVisibility(1);

  // Set the range of the lookup table
  double range[2];
  scalars->GetRange(range);
  wavefield_mapper->SetScalarRange(range[0], range[1]);
  lut->SetRange(range[0], range[1]);

  // set color gradient from white to black
  for (int i = 0; i < 256; ++i) {
    double t = static_cast<double>(i) / 255.0;
    double transparency = sigmoid(t);
    lut->SetTableValue(i, 1.0 - t, 1.0 - t, 1.0 - t, transparency);
  }

  // Create an actor
  auto wavefield_actor = vtkSmartPointer<vtkActor>::New();
  wavefield_actor->SetMapper(wavefield_mapper);

  // Create renderer
  renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(material_actor);
  renderer->AddActor(wavefield_actor);
  renderer->SetBackground(colors->GetColor3d("White").GetData());

  // Plot edges
  if (false) {
    // Create edges extractor and actors
    vtkSmartPointer<vtkExtractEdges> edges =
        vtkSmartPointer<vtkExtractEdges>::New();
    edges->SetInputData(unstructured_grid);
    edges->Update();

    vtkSmartPointer<vtkPolyDataMapper> outlineMapper =
        vtkSmartPointer<vtkPolyDataMapper>::New();
    outlineMapper->SetInputConnection(edges->GetOutputPort());
    outlineMapper->ScalarVisibilityOff();

    outlineActor = vtkSmartPointer<vtkActor>::New();
    outlineActor->SetMapper(outlineMapper);
    outlineActor->GetProperty()->SetColor(
        colors->GetColor3d("Black").GetData());
    outlineActor->GetProperty()->SetLineWidth(0.5);

    renderer->AddActor(outlineActor);
  }

  // Create render window
  render_window = vtkSmartPointer<vtkRenderWindow>::New();
  render_window->AddRenderer(renderer);
  render_window->SetSize(2560, 2560);
  render_window->SetWindowName("Wavefield");

  // Create render window interactor if on-screen
  if (output_format == specfem::display::format::on_screen) {
    render_window_interactor =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    render_window_interactor->SetRenderWindow(render_window);
  } else {
    render_window->SetOffScreenRendering(1);
  }
}

void specfem::periodic_tasks::plot_wavefield::run(
    specfem::compute::assembly &assembly, const int istep) {

  // Update the wavefield scalars only
  auto scalars = compute_wavefield_scalars(assembly);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  // Get range of scalar values
  double range[2];
  scalars->GetRange(range);
  wavefield_mapper->SetScalarRange(range[0], range[1]);

  // Update lookup table range
  lut->SetRange(range[0], range[1]);
  lut->Build();

  // Render
  render_window->Render();

  // Save image if not on-screen
  if (output_format != specfem::display::format::on_screen) {
    auto image_filter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    image_filter->SetInput(render_window);
    image_filter->Update();

    if (output_format == specfem::display::format::PNG) {
      const auto filename =
          output_folder /
          ("wavefield" + specfem::utilities::to_zero_lead(istep, 6) + ".png");
      auto writer = vtkSmartPointer<vtkPNGWriter>::New();
      writer->SetFileName(filename.string().c_str());
      writer->SetInputConnection(image_filter->GetOutputPort());
      writer->Write();
      std::string message = "Wrote wavefield image to " + filename.string();
      mpi->cout(message);
    } else if (output_format == specfem::display::format::JPG) {
      const auto filename =
          output_folder /
          ("wavefield" + specfem::utilities::to_zero_lead(istep, 6) + ".jpg");
      auto writer = vtkSmartPointer<vtkJPEGWriter>::New();
      writer->SetFileName(filename.string().c_str());
      writer->SetInputConnection(image_filter->GetOutputPort());
      writer->Write();
      std::string message = "Wrote wavefield image to " + filename.string();
      mpi->cout(message);
    } else {
      throw std::runtime_error("Unsupported output format");
    }
  }
}

void specfem::periodic_tasks::plot_wavefield::finalize(
    specfem::compute::assembly &assembly) {
  // If interactive, start the event loop if it hasn't been started
  if (output_format == specfem::display::format::on_screen &&
      render_window_interactor) {
    render_window_interactor->Start();
  }

  // Clean up VTK objects
  // VTK smart pointers will handle most of the cleanup automatically
  // But we can explicitly reset them here
  renderer = nullptr;
  render_window = nullptr;
  render_window_interactor = nullptr;
  material_actor = nullptr;
  actor = nullptr;
  outlineActor = nullptr;
  material_mapper = nullptr;
  unstructured_grid = nullptr;
  lut = nullptr;
  colors = nullptr;
}

#endif // NO_VTK
