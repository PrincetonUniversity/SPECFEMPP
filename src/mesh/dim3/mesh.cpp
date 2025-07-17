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

  int nacoustic = this->elements_types.nacoustic;
  int nelastic = this->elements_types.nelastic;
  int nporoelastic = this->elements_types.nporoelastic;

  // Print Mapping parameters
  message << "3D Mesh information :\n"
          << "------------------------------\n"
          << "Total number of spectral elements: " << nspec << "\n"
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
          << "------------------------------\n";

  return message.str();
}
