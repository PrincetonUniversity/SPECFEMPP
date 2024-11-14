
#include "writer/plot_wavefield.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"

#ifdef NO_VTK

#include <sstream>

#else

#include <boost/filesystem.hpp>
#include <vtkActor.h>
#include <vtkBiQuadraticQuad.h>
#include <vtkContourFilter.h>
#include <vtkFloatArray.h>
#include <vtkJPEGWriter.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWindowToImageFilter.h>

#endif // NO_VTK

#ifdef NO_VTK

void specfem::writer::plot_wavefield::write() {
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
    if (display_component == specfem::display::wavefield::displacement_x ||
        display_component == specfem::display::wavefield::displacement_z) {
      return specfem::wavefield::component::displacement;
    } else if (display_component == specfem::display::wavefield::velocity_x ||
               display_component == specfem::display::wavefield::velocity_z) {
      return specfem::wavefield::component::velocity;
    } else if (display_component ==
                   specfem::display::wavefield::acceleration_x ||
               display_component ==
                   specfem::display::wavefield::acceleration_z) {
      return specfem::wavefield::component::acceleration;
    } else {
      throw std::runtime_error("Unsupported component");
    }
  }();

  const int plot_index = [&display_component]() {
    if (display_component == specfem::display::wavefield::displacement_x ||
        display_component == specfem::display::wavefield::velocity_x ||
        display_component == specfem::display::wavefield::acceleration_x) {
      return 0;
    } else if (display_component ==
                   specfem::display::wavefield::displacement_z ||
               display_component == specfem::display::wavefield::velocity_z ||
               display_component ==
                   specfem::display::wavefield::acceleration_z) {
      return 1;
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

  auto points = vtkSmartPointer<vtkPoints>::New();
  points->SetNumberOfPoints(ncells * 9);
  auto scalars = vtkSmartPointer<vtkFloatArray>::New();
  scalars->SetName("Wavefield");
  scalars->SetNumberOfTuples(ncells * 9);

  auto unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();

  for (int icell = 0; icell < ncells; ++icell) {
    auto biquad = vtkSmartPointer<vtkBiQuadraticQuad>::New();
    // Bottom Corner
    points->SetPoint(icell * ncells + 0, coordinates(icell, 0, 0, 0), 0.0,
                     coordinates(icell, 0, 0, 1));
    scalars->SetTuple1(icell * ncells + 0, wavefield(icell, 0, 0, plot_index));
    biquad->GetPointIds()->SetId(0, icell * ncells + 0);

    // Bottom Right
    points->SetPoint(icell * ncells + 1, coordinates(icell, 0, ngllx - 1, 0),
                     0.0, coordinates(icell, 0, ngllx - 1, 1));
    scalars->SetTuple1(icell * ncells + 1,
                       wavefield(icell, 0, ngllx - 1, plot_index));
    biquad->GetPointIds()->SetId(1, icell * ncells + 1);

    // Top Right
    points->SetPoint(icell * ncells + 2,
                     coordinates(icell, ngllz - 1, ngllx - 1, 0), 0.0,
                     coordinates(icell, ngllz - 1, ngllx - 1, 1));
    scalars->SetTuple1(icell * ncells + 2,
                       wavefield(icell, ngllz - 1, ngllx - 1, plot_index));
    biquad->GetPointIds()->SetId(2, icell * ncells + 2);

    // Top Left
    points->SetPoint(icell * ncells + 3, coordinates(icell, ngllz - 1, 0, 0),
                     0.0, coordinates(icell, ngllz - 1, 0, 1));
    scalars->SetTuple1(icell * ncells + 3,
                       wavefield(icell, ngllz - 1, 0, plot_index));
    biquad->GetPointIds()->SetId(3, icell * ncells + 3);

    // Bottom middle
    points->SetPoint(icell * ncells + 4, coordinates(icell, 0, ngllx / 2, 0),
                     0.0, coordinates(icell, 0, ngllx / 2, 1));
    scalars->SetTuple1(icell * ncells + 4,
                       wavefield(icell, 0, ngllx / 2, plot_index));
    biquad->GetPointIds()->SetId(4, icell * ncells + 4);

    // Right middle
    points->SetPoint(icell * ncells + 5,
                     coordinates(icell, ngllz / 2, ngllx - 1, 0), 0.0,
                     coordinates(icell, ngllz / 2, ngllx - 1, 1));
    scalars->SetTuple1(icell * ncells + 5,
                       wavefield(icell, ngllz / 2, ngllx - 1, plot_index));
    biquad->GetPointIds()->SetId(5, icell * ncells + 5);

    // Top middle
    points->SetPoint(icell * ncells + 6,
                     coordinates(icell, ngllz - 1, ngllx / 2, 0), 0.0,
                     coordinates(icell, ngllz - 1, ngllx / 2, 1));
    scalars->SetTuple1(icell * ncells + 6,
                       wavefield(icell, ngllz - 1, ngllx / 2, plot_index));
    biquad->GetPointIds()->SetId(6, icell * ncells + 6);

    // Left middle
    points->SetPoint(icell * ncells + 7, coordinates(icell, ngllz / 2, 0, 0),
                     0.0, coordinates(icell, ngllz / 2, 0, 1));
    scalars->SetTuple1(icell * ncells + 7,
                       wavefield(icell, ngllz / 2, 0, plot_index));
    biquad->GetPointIds()->SetId(7, icell * ncells + 7);

    // Center
    points->SetPoint(icell * ncells + 8,
                     coordinates(icell, ngllz / 2, ngllx / 2, 0), 0.0,
                     coordinates(icell, ngllz / 2, ngllx / 2, 1));
    scalars->SetTuple1(icell * ncells + 8,
                       wavefield(icell, ngllz / 2, ngllx / 2, plot_index));
    biquad->GetPointIds()->SetId(8, icell * ncells + 8);
    unstructured_grid->InsertNextCell(biquad->GetCellType(),
                                      biquad->GetPointIds());
  }

  unstructured_grid->SetPoints(points);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  return unstructured_grid;
}
} // namespace

void specfem::writer::plot_wavefield::write() {

  const auto unstructured_grid = get_wavefield_on_vtk_grid(
      this->assembly, this->wavefield, this->component);

  // Plot a contour plot of the wavefield
  auto contour = vtkSmartPointer<vtkContourFilter>::New();
  contour->SetInputData(unstructured_grid);
  contour->GenerateValues(10, 0.0, 1.0);

  // Create a mapper
  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(contour->GetOutputPort());
  mapper->ScalarVisibilityOn();

  // Create an actor
  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  // Create a renderer
  auto renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(actor);
  renderer->SetBackground(1.0, 1.0, 1.0);

  // Create a render window
  auto render_window = vtkSmartPointer<vtkRenderWindow>::New();
  render_window->AddRenderer(renderer);
  render_window->SetSize(800, 800);

  // Create an interactor
  auto image_filter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  image_filter->SetInput(render_window);

  // Save the plot
  if (this->output_format == specfem::display::format::PNG) {
    const auto filename = this->output_folder / "wavefield.png";
    auto writer = vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(filename.string().c_str());
    writer->SetInputConnection(image_filter->GetOutputPort());
    writer->Write();
  } else if (this->output_format == specfem::display::format::JPG) {
    const auto filename = this->output_folder / "wavefield.jpg";
    auto writer = vtkSmartPointer<vtkJPEGWriter>::New();
    writer->SetFileName(filename.string().c_str());
    writer->SetInputConnection(image_filter->GetOutputPort());
    writer->Write();
  } else {
    throw std::runtime_error("Unsupported output format");
  }
}

#endif // NO_VTK
