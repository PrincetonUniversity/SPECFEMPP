
#include "periodic_tasks/plot_wavefield.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"

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
#ifdef __APPLE__
#include <vtkCocoaRenderWindow.h>
#else
#include <vtkOpenGLRenderWindow.h>
#endif
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

void specfem::periodic_tasks::plot_wavefield::run() {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

#else

namespace {

// Convert integer to string with zero leading
std::string to_zero_lead(const int value, const int n_zero) {
  auto old_str = std::to_string(value);
  int n_zero_fix =
      n_zero - std::min(n_zero, static_cast<int>(old_str.length()));
  auto new_str = std::string(n_zero_fix, '0') + old_str;
  return new_str;
}

// Sigmoid function centered at 0.0
double sigmoid(double x) { return (1 / (1 + std::exp(-100 * x)) - 0.5) * 1.5; }

// Maps different materials to different colors
vtkSmartPointer<vtkDataSetMapper>
map_materials_with_color(const specfem::compute::assembly &assembly) {

  const auto &element_types = assembly.element_types;

  const std::unordered_map<specfem::element::medium_tag, std::array<int, 3> >
      material_colors = {
        { specfem::element::medium_tag::elastic_sv, // sienna color
          { 160, 82, 45 } },
        { specfem::element::medium_tag::acoustic, // aqua color
          { 0, 255, 255 } },
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

vtkSmartPointer<vtkUnstructuredGrid> get_wavefield_on_vtk_grid(
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
} // namespace

void specfem::periodic_tasks::plot_wavefield::run() {

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

  const auto unstructured_grid = get_wavefield_on_vtk_grid(
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
          ("wavefield" + to_zero_lead(this->m_istep, 6) + ".png");
      auto writer = vtkSmartPointer<vtkPNGWriter>::New();
      writer->SetFileName(filename.string().c_str());
      writer->SetInputConnection(image_filter->GetOutputPort());
      writer->Write();
    } else if (this->output_format == specfem::display::format::JPG) {
      const auto filename =
          this->output_folder /
          ("wavefield" + std::to_string(this->m_istep) + ".jpg");
      auto writer = vtkSmartPointer<vtkJPEGWriter>::New();
      writer->SetFileName(filename.string().c_str());
      writer->SetInputConnection(image_filter->GetOutputPort());
      writer->Write();
    } else {
      throw std::runtime_error("Unsupported output format");
    }
  } else {
// Create a render window interactor
#ifdef __APPLE__
    auto render_window = vtkSmartPointer<vtkCocoaRenderWindow>::New();
#else
    auto render_window = vtkSmartPointer<vtkOpenGLRenderWindow>::New();
#endif
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
