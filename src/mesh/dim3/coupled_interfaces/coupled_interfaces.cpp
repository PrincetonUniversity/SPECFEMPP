#include "mesh/mesh.hpp"
#include <iostream>

void specfem::mesh::coupled_interfaces<specfem::dimension::type::dim3>::print()
    const {

  if (acoustic_elastic) {
    std::cout << "Acoustic Elastic Interface Metadata:" << std::endl;
    std::cout << "================================================"
              << std::endl;
    std::cout << "  ngllsquare:............... " << ngllsquare << std::endl;
    std::cout << "  num_coupling_ac_el_faces:. " << num_coupling_ac_el_faces
              << std::endl;

    // Print the acoustic elastic interface metadata
    std::cout << "  Array sizes:" << std::endl;
    std::cout << "  -----------------------------------------------"
              << std::endl;
    std::cout << "  ispec:.................. "
              << acoustic_elastic_interface.ispec.extent(0) << std::endl;
    std::cout << "  ijk:.................... "
              << acoustic_elastic_interface.ijk.extent(0) << " "
              << acoustic_elastic_interface.ijk.extent(1) << " "
              << acoustic_elastic_interface.ijk.extent(2) << std::endl;
    std::cout << "  jacobian2Dw:............ "
              << acoustic_elastic_interface.jacobian2Dw.extent(0) << " "
              << acoustic_elastic_interface.jacobian2Dw.extent(1) << std::endl;
    std::cout << "  normal:................. "
              << acoustic_elastic_interface.normal.extent(0) << " "
              << acoustic_elastic_interface.normal.extent(1) << " "
              << acoustic_elastic_interface.normal.extent(2) << std::endl;
  } else {
    std::cout << "No acoustic elastic interfaces" << std::endl;
  }

  if (acoustic_poroelastic) {

    std::cout << "Acoustic Poroelastic Interface Metadata:" << std::endl;
    std::cout << "================================================"
              << std::endl;
    std::cout << "  ngllsquare:............... " << ngllsquare << std::endl;
    std::cout << "  num_coupling_ac_po_faces:. " << num_coupling_ac_po_faces
              << std::endl;

    // Print the acoustic poroelastic interface metadata
    std::cout << "  Array sizes:" << std::endl;
    std::cout << "  -----------------------------------------------"
              << std::endl;
    std::cout << "  ispec:.................. "
              << acoustic_poroelastic_interface.ispec.extent(0) << std::endl;
    std::cout << "  ijk:.................... "
              << acoustic_poroelastic_interface.ijk.extent(0) << " "
              << acoustic_poroelastic_interface.ijk.extent(1) << " "
              << acoustic_poroelastic_interface.ijk.extent(2) << std::endl;
    std::cout << "  jacobian2Dw:............ "
              << acoustic_poroelastic_interface.jacobian2Dw.extent(0) << " "
              << acoustic_poroelastic_interface.jacobian2Dw.extent(1)
              << std::endl;
    std::cout << "  normal:................. "
              << acoustic_poroelastic_interface.normal.extent(0) << " "
              << acoustic_poroelastic_interface.normal.extent(1) << " "
              << acoustic_poroelastic_interface.normal.extent(2) << std::endl;
  } else {
    std::cout << "No acoustic poroelastic interfaces" << std::endl;
  }

  if (elastic_poroelastic) {

    std::cout << "Elastic Poroelastic Interface Metadata:" << std::endl;
    std::cout << "================================================"
              << std::endl;
    std::cout << "  ngllsquare:............... " << ngllsquare << std::endl;
    std::cout << "  num_coupling_el_po_faces:. " << num_coupling_el_po_faces
              << std::endl;

    // Print the elastic poroelastic interface metadata
    std::cout << "  Array sizes:" << std::endl;
    std::cout << "  -----------------------------------------------"
              << std::endl;
    std::cout << "  ispec:.................. "
              << elastic_poroelastic_interface.ispec.extent(0) << std::endl;
    std::cout << "  ijk:.................... "
              << elastic_poroelastic_interface.ijk.extent(0) << " "
              << elastic_poroelastic_interface.ijk.extent(1) << " "
              << elastic_poroelastic_interface.ijk.extent(2) << std::endl;
    std::cout << "  jacobian2Dw:............ "
              << elastic_poroelastic_interface.jacobian2Dw.extent(0) << " "
              << elastic_poroelastic_interface.jacobian2Dw.extent(1)
              << std::endl;
    std::cout << "  normal:................. "
              << elastic_poroelastic_interface.normal.extent(0) << " "
              << elastic_poroelastic_interface.normal.extent(1) << " "
              << elastic_poroelastic_interface.normal.extent(2) << std::endl;

  } else {
    std::cout << "No elastic poroelastic interfaces" << std::endl;
  }
}
