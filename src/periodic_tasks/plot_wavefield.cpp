
#include "periodic_tasks/plot_wavefield.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"
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

void specfem::periodic_tasks::plot_wavefield::run(
    specfem::compute::assembly &assembly, const int istep) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

#else

namespace {

// Sigmoid function centered at 0.0
double sigmoid(double x) { return (1 / (1 + std::exp(-100 * x)) - 0.5) * 1.5; }

// Maps different materials to different colors
vtkSmartPointer<vtkDataSetMapper>
map_materials_with_color(const specfem::compute::assembly &assembly) {

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
vtkSmartPointer<vtkUnstructuredGrid> get_wavefield_on_vtk_biquad_grid(
    specfem::compute::assembly &assembly,
    const specfem::wavefield::simulation_field type,
    const specfem::display::wavefield &display_component) {

  const auto component = [&display_component]() {
    if (display_component == specfem::display::wavefield::displacement) {
      return specfem::wavefield::type::displacement;
    } else if (display_component == specfem::display::wavefield::velocity) {
      return specfem::wavefield::type::velocity;
    } else if (display_component == specfem::display::wavefield::acceleration) {
      return specfem::wavefield::type::acceleration;
    } else if (display_component == specfem::display::wavefield::pressure) {
      return specfem::wavefield::type::pressure;
    } else {
      throw std::runtime_error("Unsupported component");
    }
  }();

  const auto &wavefield =
      assembly.generate_wavefield_on_entire_grid(type, component);
  const auto &coordinates = assembly.mesh.points.h_coord;

  const int ncells = wavefield.extent(0);
  const int ngllz = wavefield.extent(1);
  const int ngllx = wavefield.extent(2);

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
  auto scalars = vtkSmartPointer<vtkFloatArray>::New();

  for (int icell = 0; icell < ncells; ++icell) {
    for (int i = 0; i < cell_points; ++i) {
      points->InsertNextPoint(coordinates(0, icell, z_index[i], x_index[i]),
                              coordinates(1, icell, z_index[i], x_index[i]),
                              0.0);
      if (component == specfem::wavefield::type::pressure) {
        scalars->InsertNextValue(
            std::abs(wavefield(icell, z_index[i], x_index[i], 0)));
      } else {
        scalars->InsertNextValue(
            std::sqrt((wavefield(icell, z_index[i], x_index[i], 0) *
                       wavefield(icell, z_index[i], x_index[i], 0)) +
                      (wavefield(icell, z_index[i], x_index[i], 1) *
                       wavefield(icell, z_index[i], x_index[i], 1))));
      }
    }
    auto quad = vtkSmartPointer<vtkBiQuadraticQuad>::New();
    for (int i = 0; i < cell_points; ++i) {
      quad->GetPointIds()->SetId(i, icell * cell_points + i);
    }
    cells->InsertNextCell(quad);
  }

  auto unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  unstructured_grid->SetPoints(points);
  unstructured_grid->SetCells(VTK_BIQUADRATIC_QUAD, cells);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  return unstructured_grid;
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
vtkSmartPointer<vtkUnstructuredGrid> get_wavefield_on_vtk_quad_grid(
    specfem::compute::assembly &assembly,
    const specfem::wavefield::simulation_field type,
    const specfem::display::wavefield &display_component) {

  const auto component = [&display_component]() {
    if (display_component == specfem::display::wavefield::displacement) {
      return specfem::wavefield::type::displacement;
    } else if (display_component == specfem::display::wavefield::velocity) {
      return specfem::wavefield::type::velocity;
    } else if (display_component == specfem::display::wavefield::acceleration) {
      return specfem::wavefield::type::acceleration;
    } else if (display_component == specfem::display::wavefield::pressure) {
      return specfem::wavefield::type::pressure;
    } else {
      throw std::runtime_error("Unsupported component");
    }
  }();

  const auto &wavefield =
      assembly.generate_wavefield_on_entire_grid(type, component);
  const auto &coordinates = assembly.mesh.points.h_coord;

  const int nspec = wavefield.extent(0);
  const int ngllz = wavefield.extent(1);
  const int ngllx = wavefield.extent(2);

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
  auto scalars = vtkSmartPointer<vtkFloatArray>::New();

  int point_counter = 0; // Keep track of the global point index

  // Loop over the cells
  for (int ispec = 0; ispec < nspec; ++ispec) {
    for (int iz = 0; iz < ngllz - 1; ++iz) {
      for (int ix = 0; ix < ngllx - 1; ++ix) {

        const int face_index =
            ispec * (ngllz - 1) * (ngllx - 1) + iz * (ngllx - 1) + ix;

        auto quad = vtkSmartPointer<vtkQuad>::New();

        for (int ipoint = 0; ipoint < n_cell_points; ++ipoint) {
          int iz_pos = iz + z_index[ipoint];
          int ix_pos = ix + x_index[ipoint];

          // Insert the point
          points->InsertNextPoint(coordinates(0, ispec, iz_pos, ix_pos),
                                  coordinates(1, ispec, iz_pos, ix_pos), 0.0);

          // Insert scalar value
          if (component == specfem::wavefield::type::pressure) {
            scalars->InsertNextValue(
                std::abs(wavefield(ispec, iz_pos, ix_pos, 0)));
          } else {
            scalars->InsertNextValue(
                std::sqrt((wavefield(ispec, iz_pos, ix_pos, 0) *
                           wavefield(ispec, iz_pos, ix_pos, 0)) +
                          (wavefield(ispec, iz_pos, ix_pos, 1) *
                           wavefield(ispec, iz_pos, ix_pos, 1))));
          }

          // Set the point ID for this quad
          quad->GetPointIds()->SetId(ipoint, point_counter);
          point_counter++;
        }

        // Add the cell
        cells->InsertNextCell(quad);
      }
    }
  }

  auto unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  unstructured_grid->SetPoints(points);
  unstructured_grid->SetCells(VTK_QUAD, cells);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  return unstructured_grid;
}
} // namespace

void specfem::periodic_tasks::plot_wavefield::run(
    specfem::compute::assembly &assembly, const int istep) {

  auto colors = vtkSmartPointer<vtkNamedColors>::New();

  if (this->output_format != specfem::display::format::on_screen) {
    vtkSmartPointer<vtkGraphicsFactory> graphics_factory;
    graphics_factory->SetOffScreenOnlyMode(1);
    graphics_factory->SetUseMesaClasses(1);
  }

  auto material_mapper = map_materials_with_color(this->assembly);

  // Create an actor
  auto material_actor = vtkSmartPointer<vtkActor>::New();
  material_actor->SetMapper(material_mapper);

  const auto unstructured_grid = get_wavefield_on_vtk_quad_grid(
      this->assembly, this->wavefield, this->component);
  const int ncell = unstructured_grid->GetNumberOfCells();

  double range[2];

  unstructured_grid->GetPointData()->GetScalars()->GetRange(range);

  // create a lookup table to map cell data to colors. The range is from
  // range[0] to range[1]
  vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
  lut->SetNumberOfTableValues(256);
  lut->SetRange(range[0], range[1]);
  lut->Build();

  // set color gradient from white to black
  for (int i = 0; i < 256; ++i) {
    double t = static_cast<double>(i) / 255.0;
    double transparency = sigmoid(t);
    lut->SetTableValue(i, 1.0 - t, 1.0 - t, 1.0 - t, transparency);
  }

  // Create a mapper
  auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
  mapper->SetInputData(unstructured_grid);
  mapper->SetLookupTable(lut);
  mapper->SetScalarRange(range[0], range[1]);
  mapper->SetScalarModeToUsePointData();
  mapper->SetColorModeToMapScalars();
  mapper->SetScalarVisibility(1);

  // Create an actor
  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  vtkSmartPointer<vtkExtractEdges> edges =
      vtkSmartPointer<vtkExtractEdges>::New();
  edges->SetInputData(unstructured_grid);
  edges->Update();

  vtkSmartPointer<vtkPolyDataMapper> outlineMapper =
      vtkSmartPointer<vtkPolyDataMapper>::New();
  outlineMapper->SetInputConnection(edges->GetOutputPort());
  outlineMapper->ScalarVisibilityOff();

  vtkSmartPointer<vtkActor> outlineActor = vtkSmartPointer<vtkActor>::New();
  outlineActor->SetMapper(outlineMapper);
  outlineActor->GetProperty()->SetColor(colors->GetColor3d("Black").GetData());
  outlineActor->GetProperty()->SetLineWidth(0.5);

  // Create a renderer
  auto renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(material_actor);
  // renderer->AddActor(outlineActor);
  renderer->AddActor(actor);
  renderer->SetBackground(colors->GetColor3d("White").GetData());
  renderer->ResetCamera();

  if (this->output_format != specfem::display::format::on_screen) {
    // Create a render window
    auto render_window = vtkSmartPointer<vtkRenderWindow>::New();
    render_window->SetOffScreenRendering(1);
    render_window->AddRenderer(renderer);
    render_window->SetSize(2560, 2560);
    render_window->SetWindowName("Wavefield");
    auto image_filter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    image_filter->SetInput(render_window);
    image_filter->Update();

    // Save the plot
    if (this->output_format == specfem::display::format::PNG) {
      const auto filename =
          this->output_folder /
          ("wavefield" + specfem::utilities::to_zero_lead(istep, 6) + ".png");
      auto writer = vtkSmartPointer<vtkPNGWriter>::New();
      writer->SetFileName(filename.string().c_str());
      writer->SetInputConnection(image_filter->GetOutputPort());
      writer->Write();
    } else if (this->output_format == specfem::display::format::JPG) {
      const auto filename =
          this->output_folder /
          ("wavefield" + specfem::utilities::to_zero_lead(istep, 6) + ".jpg");
      auto writer = vtkSmartPointer<vtkJPEGWriter>::New();
      writer->SetFileName(filename.string().c_str());
      writer->SetInputConnection(image_filter->GetOutputPort());
      writer->Write();
    } else {
      throw std::runtime_error("Unsupported output format");
    }
  } else {
    // Create a render window interactor
    auto render_window = vtkSmartPointer<vtkRenderWindow>::New();
    render_window->AddRenderer(renderer);
    render_window->SetSize(2560, 2560);
    render_window->SetWindowName("Wavefield");

    auto render_window_interactor =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    render_window_interactor->SetRenderWindow(render_window);

    // Start the event loop
    render_window->Render();
    render_window_interactor->Start();
  }
}

#endif // NO_VTK
