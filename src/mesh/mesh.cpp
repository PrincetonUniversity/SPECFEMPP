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

std::string specfem::mesh::mesh<specfem::dimension::type::dim2>::print() const {

  int n_elastic;
  int n_acoustic;

  Kokkos::parallel_reduce(
      "specfem::mesh::mesh::print", specfem::kokkos::HostRange(0, this->nspec),
      KOKKOS_CLASS_LAMBDA(const int ispec, int &n_elastic, int &n_acoustic) {
        if (this->materials.material_index_mapping(ispec).type ==
            specfem::element::medium_tag::elastic) {
          n_elastic++;
        } else if (this->materials.material_index_mapping(ispec).type ==
                   specfem::element::medium_tag::acoustic) {
          n_acoustic++;
        }
      },
      n_elastic, n_acoustic);

  std::ostringstream message;

  message
      << "Spectral element information:\n"
      << "------------------------------\n"
      << "Total number of spectral elements : " << this->nspec << "\n"
      << "Total number of spectral elements assigned to elastic material : "
      << n_elastic << "\n"
      << "Total number of spectral elements assigned to acoustic material : "
      << n_acoustic << "\n"
      << "Total number of geometric points : " << this->npgeo << "\n";

  return message.str();
}
