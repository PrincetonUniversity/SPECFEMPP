#include "mesh/mesh.hpp"
#include <iostream>

std::string
specfem::mesh::coupled_interfaces<specfem::dimension::type::dim3>::print()
    const {

  std::ostringstream message;

  if (acoustic_elastic) {
    message << "Acoustic Elastic Interface Metadata:"
            << "\n";
    message << "================================================"
            << "\n";
    message << "  ngllsquare:............... " << ngllsquare << "\n";
    message << "  num_coupling_ac_el_faces:. " << num_coupling_ac_el_faces
            << "\n";

    // Print the acoustic elastic interface metadata
    message << "  Array sizes:"
            << "\n";
    message << "  -----------------------------------------------"
            << "\n";
    message << "  ispec:.................. "
            << acoustic_elastic_interface.ispec.extent(0) << "\n";
    message << "  ijk:.................... "
            << acoustic_elastic_interface.ijk.extent(0) << " "
            << acoustic_elastic_interface.ijk.extent(1) << " "
            << acoustic_elastic_interface.ijk.extent(2) << "\n";
    message << "  jacobian2Dw:............ "
            << acoustic_elastic_interface.jacobian2Dw.extent(0) << " "
            << acoustic_elastic_interface.jacobian2Dw.extent(1) << "\n";
    message << "  normal:................. "
            << acoustic_elastic_interface.normal.extent(0) << " "
            << acoustic_elastic_interface.normal.extent(1) << " "
            << acoustic_elastic_interface.normal.extent(2) << "\n";
  } else {
    message << "No acoustic elastic interfaces"
            << "\n";
  }

  if (acoustic_poroelastic) {

    message << "Acoustic Poroelastic Interface Metadata:"
            << "\n";
    message << "================================================"
            << "\n";
    message << "  ngllsquare:............... " << ngllsquare << "\n";
    message << "  num_coupling_ac_po_faces:. " << num_coupling_ac_po_faces
            << "\n";

    // Print the acoustic poroelastic interface metadata
    message << "  Array sizes:"
            << "\n";
    message << "  -----------------------------------------------"
            << "\n";
    message << "  ispec:.................. "
            << acoustic_poroelastic_interface.ispec.extent(0) << "\n";
    message << "  ijk:.................... "
            << acoustic_poroelastic_interface.ijk.extent(0) << " "
            << acoustic_poroelastic_interface.ijk.extent(1) << " "
            << acoustic_poroelastic_interface.ijk.extent(2) << "\n";
    message << "  jacobian2Dw:............ "
            << acoustic_poroelastic_interface.jacobian2Dw.extent(0) << " "
            << acoustic_poroelastic_interface.jacobian2Dw.extent(1) << "\n";
    message << "  normal:................. "
            << acoustic_poroelastic_interface.normal.extent(0) << " "
            << acoustic_poroelastic_interface.normal.extent(1) << " "
            << acoustic_poroelastic_interface.normal.extent(2) << "\n";
  } else {
    message << "No acoustic poroelastic interfaces"
            << "\n";
  }

  if (elastic_poroelastic) {

    message << "Elastic Poroelastic Interface Metadata:"
            << "\n";
    message << "================================================"
            << "\n";
    message << "  ngllsquare:............... " << ngllsquare << "\n";
    message << "  num_coupling_el_po_faces:. " << num_coupling_el_po_faces
            << "\n";

    // Print the elastic poroelastic interface metadata
    message << "  Array sizes:"
            << "\n";
    message << "  -----------------------------------------------"
            << "\n";
    message << "  ispec:.................. "
            << elastic_poroelastic_interface.ispec.extent(0) << "\n";
    message << "  ijk:.................... "
            << elastic_poroelastic_interface.ijk.extent(0) << " "
            << elastic_poroelastic_interface.ijk.extent(1) << " "
            << elastic_poroelastic_interface.ijk.extent(2) << "\n";
    message << "  jacobian2Dw:............ "
            << elastic_poroelastic_interface.jacobian2Dw.extent(0) << " "
            << elastic_poroelastic_interface.jacobian2Dw.extent(1) << "\n";
    message << "  normal:................. "
            << elastic_poroelastic_interface.normal.extent(0) << " "
            << elastic_poroelastic_interface.normal.extent(1) << " "
            << elastic_poroelastic_interface.normal.extent(2) << "\n";

  } else {
    message << "No elastic poroelastic interfaces"
            << "\n";
  }

  return message.str();
}
