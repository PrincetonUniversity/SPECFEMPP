#include "mesh/mesh.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "medium/material.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

std::string specfem::mesh::mesh<specfem::dimension::type::dim3>::print() const {

  std::ostringstream message;

  int nspec = this->parameters.nspec;
  int nglob = this->parameters.nglob;
  int nspec_irregular = this->parameters.nspec_irregular;
  int ngllx = this->parameters.ngllx;
  int nglly = this->parameters.nglly;
  int ngllz = this->parameters.ngllz;

  int nacoustic = this->element_types.nacoustic;
  int nelastic = this->element_types.nelastic;
  int nporoelastic = this->element_types.nporoelastic;

  const auto bbox = this->coordinates.bounding_box();

  // Print Mapping parameters
  message << "3D Mesh information :\n"
          << "------------------------------\n"
          << "Total number of spectral elements: " << nspec << "\n"
          << "Total number of distinct control nodes: "
          << this->control_nodes.nnodes << "\n"
          << "Total number of global nodes: " << nglob << "\n"
          << "Total number of irregular spectral elements: " << nspec_irregular
          << "\n"
          << "Total number of GLLX: " << ngllx << "\n"
          << "Total number of GLLY: " << nglly << "\n"
          << "Total number of GLLZ: " << ngllz << "\n"
          << "\n"
          << "Total number of acoustic spectral elements: " << nacoustic << "\n"
          << "Total number of elastic spectral elements: " << nelastic << "\n"
          << "Total number of poroelastic spectral elements: " << nporoelastic
          << "\n"
          << "Bounding box: xmin/xmax: " << bbox[0] << " / " << bbox[1] << "\n"
          << "Bounding box: ymin/ymax: " << bbox[2] << " / " << bbox[3] << "\n"
          << "Bounding box: zmin/zmax: " << bbox[4] << " / " << bbox[5] << "\n"
          << "------------------------------\n"
          << "Parameters:\n"
          << this->parameters.print() << "------------------------------\n";

  return message.str();
}
