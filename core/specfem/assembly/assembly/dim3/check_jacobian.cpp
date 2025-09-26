#include "specfem/assembly/assembly.hpp"

void specfem::assembly::assembly<
    specfem::dimension::type::dim3>::check_jacobian_matrix() const {
  const auto [found, small_jacobian] =
      this->jacobian_matrix.check_small_jacobian();

  if (found) {
    // #ifdef NO_VTK
    std::ostringstream message;
    message
        << "Small Jacobian found within the mesh. Please refine the mesh.\n"
        << "To generate a plot of the mesh with small Jacobian, please build "
           "SPECFEM++ with VTK support.\n";
    // #else
    //     // current working directory
    //     boost::filesystem::path output_folder =
    //         boost::filesystem::current_path() / "error_jacobian";
    //     boost::filesystem::create_directories(output_folder);

    //     std::ostringstream message;
    //     message << "Small Jacobian found within the mesh. Please refine the
    //     mesh.\n"
    //             << "Generating a plot of the mesh with small Jacobian at"
    //             << output_folder.string() << "\n";

    //     plot_small_jacobian(this->mesh, small_jacobian, output_folder);

    // #endif

    throw std::runtime_error(message.str());
  }

  return;
}
