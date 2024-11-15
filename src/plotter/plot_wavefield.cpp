
#include "plotter/plot_wavefield.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"

#ifdef NO_VTK

#include <sstream>

#else

#include <boost/filesystem.hpp>
#include <vtkActor.h>
#include <vtkBiQuadraticQuad.h>
#include <vtkDataSetMapper.h>
#include <vtkExtractEdges.h>
#include <vtkFloatArray.h>
#include <vtkGraphicsFactory.h>
#include <vtkJPEGWriter.h>
#include <vtkNamedColors.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWindowToImageFilter.h>

#endif // NO_VTK

#ifdef NO_VTK

void specfem::plotter::plot_wavefield::write() {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

#else

namespace {

vtkSmartPointer<vtkUnstructuredGrid> get_wavefield_on_vtk_grid(
    specfem::compute::assembly &assembly, const specfem::wavefield::type type,
    const specfem::display::wavefield &display_component) {

  const auto component = [&display_component]() {
    if (display_component == specfem::display::wavefield::displacement) {
      return specfem::wavefield::component::displacement;
    } else if (display_component == specfem::display::wavefield::velocity) {
      return specfem::wavefield::component::velocity;
    } else if (display_component == specfem::display::wavefield::acceleration) {
      return specfem::wavefield::component::acceleration;
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
      scalars->InsertNextValue(
          std::sqrt((wavefield(icell, z_index[i], x_index[i], 0) *
                     wavefield(icell, z_index[i], x_index[i], 0)) +
                    (wavefield(icell, z_index[i], x_index[i], 1) *
                     wavefield(icell, z_index[i], x_index[i], 1))));
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

void specfem::plotter::plot_wavefield::plot() {

  auto colors = vtkSmartPointer<vtkNamedColors>::New();

  vtkSmartPointer<vtkGraphicsFactory> graphics_factory;
  graphics_factory->SetOffScreenOnlyMode(1);
  graphics_factory->SetUseMesaClasses(1);

  const auto unstructured_grid = get_wavefield_on_vtk_grid(
      this->assembly, this->wavefield, this->component);
  const int ncell = unstructured_grid->GetNumberOfCells();

  double range[2];

  unstructured_grid->GetPointData()->GetScalars()->GetRange(range);

  auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
  mapper->SetInputData(unstructured_grid);

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
  outlineActor->GetProperty()->SetLineWidth(1.0);

  // Create a renderer
  auto renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(actor);
  renderer->AddActor(outlineActor);
  renderer->SetBackground(colors->GetColor3d("White").GetData());
  renderer->ResetCamera();

  // Create a render window
  auto render_window = vtkSmartPointer<vtkRenderWindow>::New();
  render_window->SetOffScreenRendering(1);
  render_window->AddRenderer(renderer);
  render_window->SetSize(1280, 1280);
  render_window->SetWindowName("Wavefield");

  auto image_filter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  image_filter->SetInput(render_window);
  image_filter->Update();

  // Save the plot
  if (this->output_format == specfem::display::format::PNG) {
    const auto filename =
        this->output_folder /
        ("wavefield" + std::to_string(this->m_istep) + ".png");
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
}

#endif // NO_VTK
