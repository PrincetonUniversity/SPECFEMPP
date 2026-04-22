#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include <map>

specfem::assembly::assembly<specfem::element::dimension_tag::dim3>::assembly(
    const specfem::mesh::mesh<dimension_tag> &mesh,
    const specfem::quadrature::quadratures &quadratures,
    std::vector<std::shared_ptr<specfem::sources::source<dimension_tag> > >
        &sources,
    const std::vector<
        std::shared_ptr<specfem::receivers::receiver<dimension_tag> > >
        &receivers,
    const std::vector<specfem::enums::wavefield> &stypes, const type_real t0,
    const type_real dt, const int max_timesteps, const int max_sig_step,
    const int nsteps_between_samples,
    const specfem::simulation::type simulation,
    const bool allocate_boundary_values,
    const std::shared_ptr<specfem::io::reader> &property_reader) {

  const int nspec = mesh.nspec;
  const int ngllz = mesh.element_grid.ngllz;
  const int nglly = mesh.element_grid.nglly;
  const int ngllx = mesh.element_grid.ngllx;
  const int ngnod = mesh.control_nodes.ngnod;

  this->mesh = { nspec,
                 ngnod,
                 ngllz,
                 nglly,
                 ngllx,
                 mesh.tags,
                 mesh.adjacency_graph,
                 mesh.control_nodes,
                 quadratures };

  this->element_types = { nspec, ngllz, nglly, ngllx, this->mesh, mesh.tags };

  this->mpi_interfaces = { mesh.adjacency_graph, this->mesh, ngllz, nglly,
                           ngllx };

  this->element_intersections = { ngllz, nglly, ngllx, this->mesh,
                                  this->element_types };

  this->jacobian_matrix = { this->mesh };

  this->properties = { nspec, ngllz,          nglly,
                       ngllx, mesh.materials, this->element_types };

  this->kernels = { this->mesh.nspec, this->mesh.element_grid.ngllz,
                    this->mesh.element_grid.nglly,
                    this->mesh.element_grid.ngllx, this->element_types };

  this->sources = {
    sources, this->mesh, this->jacobian_matrix, this->element_types,
    t0,      dt,         max_timesteps
  };
  this->receivers = {
    max_sig_step, dt,         t0,        nsteps_between_samples, receivers,
    stypes,       this->mesh, mesh.tags, this->element_types
  };
  this->boundaries = { this->mesh.nspec,
                       this->mesh.element_grid.ngllz,
                       this->mesh.element_grid.nglly,
                       this->mesh.element_grid.ngllx,
                       mesh,
                       this->mesh,
                       this->jacobian_matrix };
  this->conforming_interfaces = {
    ngllz,     nglly, ngllx, this->element_intersections, this->jacobian_matrix,
    this->mesh
  };
  this->fields = { this->mesh, this->element_types, simulation };

  // if (allocate_boundary_values)
  //   this->boundary_values = { max_timesteps, this->mesh, this->element_types,
  //                             this->boundaries };

  // Currently done in the mesher!
  this->check_jacobian_matrix();

  this->info = { this->mesh, this->properties, this->element_types };

  return;
}

std::string
specfem::assembly::assembly<specfem::element::dimension_tag::dim3>::print()
    const {
  std::ostringstream message;
  message << "Assembly information:\n"
          << "------------------------------\n"
          << "  Total number of spectral elements             : "
          << this->mesh.nspec << "\n"
          << "  Total number of quadrature points per element : "
          << this->mesh.element_grid.ngllz << "\n"
          << this->info.string() << "\n";

  int total_elements = 0;

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC)), {
    // Getting the number of elements per medium
    int n_elements = this->element_types.get_number_of_elements(_medium_tag_);

    // Printing the number of elements if more than 0
    if (n_elements > 0) {
      // Adding the number of elements to the total
      total_elements += n_elements;

      message << "   Total number of elements of type "
              << specfem::element::to_string(_medium_tag_) << " : "
              << n_elements << "\n";
    };
  })

  if (total_elements == mesh.nspec) {
    message << "  All elements accounted for.\n";
  } else {
    message << " NOT ALL ELEMENTS ACCOUNTED FOR\n";
    message << "  Mesh elements:              " << mesh.nspec << "\n";
    message << "  Assembly elements counted:  " << total_elements << "\n";
    message << "  Total unaccounted elements: " << (mesh.nspec - total_elements)
            << "\n";
    throw std::runtime_error(message.str());
  }

#ifdef SPECFEM_ENABLE_MPI
  {
    const auto &src_islice = this->sources.get_islice();
    if (!src_islice.empty()) {
      std::map<int, std::vector<int> > sources_per_rank;
      for (int i = 0; i < static_cast<int>(src_islice.size()); ++i)
        sources_per_rank[src_islice[i]].push_back(i);
      message << "Sources per MPI rank:\n";
      for (const auto &[rank, indices] : sources_per_rank) {
        message << "  Rank " << rank << " :";
        for (int idx : indices)
          message << " " << idx;
        message << "\n";
      }
    }
    if (this->receivers.size() > 0) {
      std::map<int, std::vector<std::string> > stations_per_rank;
      size_t idx = 0;
      for (const auto &station : this->receivers.stations()) {
        int rank = this->receivers.get_receiver_islice(idx);
        stations_per_rank[rank].push_back(station.network_name + "." +
                                          station.station_name);
        ++idx;
      }
      message << "Stations per MPI rank:\n";
      for (const auto &[rank, names] : stations_per_rank) {
        message << "  Rank " << rank << " :";
        for (const auto &name : names)
          message << " " << name;
        message << "\n";
      }
    }
  }
#endif

  return message.str();
}
