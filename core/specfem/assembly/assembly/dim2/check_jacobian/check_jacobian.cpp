#include "enumerations/interface.hpp"
#include "specfem/assembly/assembly.hpp"

#ifdef NO_VTK

#include <sstream>

#else

#include <boost/filesystem.hpp>
#include <vtkActor.h>
#include <vtkCellData.h>
#include <vtkDataSetMapper.h>
#include <vtkExtractEdges.h>
#include <vtkGraphicsFactory.h>
#include <vtkNamedColors.h>
#include <vtkPNGWriter.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkQuad.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWindowToImageFilter.h>

#endif // NO_VTK

#ifndef NO_VTK

template <specfem::dimension::type DimensionTag>
std::tuple<vtkSmartPointer<vtkDataSetMapper>,
           vtkSmartPointer<vtkPolyDataMapper> >
map_small_jacobian(const specfem::assembly::mesh<DimensionTag> &mesh,
                   const Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace>
                       &small_jacobian);

// Maps different materials to different colors
template <>
std::tuple<vtkSmartPointer<vtkDataSetMapper>,
           vtkSmartPointer<vtkPolyDataMapper> >
map_small_jacobian(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace>
        &small_jacobian) {

  const std::array<int, 3> blue = { 0, 0, 255 }; // Tag all elements with small
                                                 // jacobian blue
  const std::array<int, 3> grey = { 128, 128, 128 }; // Tag all elements with
                                                     // large jacobian grey

  const int nspec = mesh.nspec;
  const int ngllx = mesh.element_grid.ngllx;
  const int ngllz = mesh.element_grid.ngllz;

  const auto &coordinates = mesh.h_coord;

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

    const auto color = (small_jacobian(icell) ? blue : grey);
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

  // Extract edges
  vtkSmartPointer<vtkExtractEdges> edges =
      vtkSmartPointer<vtkExtractEdges>::New();
  edges->SetInputData(unstructured_grid);
  edges->Update();
  vtkSmartPointer<vtkPolyDataMapper> outlineMapper =
      vtkSmartPointer<vtkPolyDataMapper>::New();
  outlineMapper->SetInputConnection(edges->GetOutputPort());

  return std::make_tuple(mapper, outlineMapper);
}

template <specfem::dimension::type DimensionTag>
void plot_small_jacobian(
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace>
        &small_jacobian,
    boost::filesystem::path output_folder);

template <>
void plot_small_jacobian(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace>
        &small_jacobian,
    boost::filesystem::path output_folder) {

  auto colors = vtkSmartPointer<vtkNamedColors>::New();

  vtkSmartPointer<vtkGraphicsFactory> graphics_factory;
  graphics_factory->SetOffScreenOnlyMode(1);
  graphics_factory->SetUseMesaClasses(1);

  auto [mapper, outline_mapper] = map_small_jacobian(mesh, small_jacobian);

  // Create an actor
  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetOpacity(0.7); // Make fill slightly transparent

  // Create an actor for the outline
  auto outlineActor = vtkSmartPointer<vtkActor>::New();
  outlineActor->SetMapper(outline_mapper);
  outlineActor->GetProperty()->SetColor(colors->GetColor3d("Black").GetData());
  outlineActor->GetProperty()->SetLineWidth(2.0);

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
  render_window->SetSize(2560, 2560);
  render_window->SetWindowName("Small Jacobian Elements");
  auto image_filter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  image_filter->SetInput(render_window);
  image_filter->Update();

  const auto filename = output_folder / "small_jacobian.png";
  auto writer = vtkSmartPointer<vtkPNGWriter>::New();
  writer->SetFileName(filename.string().c_str());
  writer->SetInputConnection(image_filter->GetOutputPort());
  writer->Write();
}

#endif // NO_VTK

void specfem::assembly::assembly<
    specfem::dimension::type::dim2>::check_jacobian_matrix() const {
  const auto [found, small_jacobian] =
      this->jacobian_matrix.check_small_jacobian();

  if (found) {

#ifdef NO_VTK
    std::ostringstream message;
    message
        << "Small Jacobian found within the mesh. Please refine the mesh.\n"
        << "To generate a plot of the mesh with small Jacobian, please build "
           "SPECFEM++ with VTK support.\n";
#else
    // current working directory
    boost::filesystem::path output_folder =
        boost::filesystem::current_path() / "error_jacobian";
    boost::filesystem::create_directories(output_folder);

    std::ostringstream message;
    message << "Small Jacobian found within the mesh. Please refine the mesh.\n"
            << "Generating a plot of the mesh with small Jacobian at"
            << output_folder.string() << "\n";

    plot_small_jacobian(this->mesh, small_jacobian, output_folder);

#endif

    throw std::runtime_error(message.str());
  }

  return;
}
