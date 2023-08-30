#ifndef _COMPUTE_COUPLED_INTERFACES_HPP
#define _COMPUTE_COUPLED_INTERFACES_HPP

#include "kokkos_abstractions.h"
#include "mesh/coupled_interfaces/coupled_interfaces.hpp"
#include "specfem_enums.hpp"

namespace specfem {
namespace compute {
namespace coupled_interfaces {
struct elastic_acoustic {
  specfem::kokkos::HostMirror1d<int> h_elastic_ispec;  ///< Index of the spectal
                                                       ///< element on the
                                                       ///< elastic side of the
                                                       ///< interface (host
                                                       ///< mirror)
  specfem::kokkos::HostMirror1d<int> h_acoustic_ispec; ///< Index of the spectal
                                                       ///< element on the
                                                       ///< acoustic side of the
                                                       ///< interface (host
                                                       ///< mirror)
  specfem::kokkos::DeviceView1d<int> elastic_ispec;    ///< Index of the spectal
                                                    ///< element on the elastic
                                                    ///< side of the interface
  specfem::kokkos::DeviceView1d<int> acoustic_ispec; ///< Index of the spectal
                                                     ///< element on the
                                                     ///< acoustic side of the
                                                     ///< interface
  specfem::kokkos::DeviceView1d<specfem::enums::coupling::edge::type>
      elastic_edge; ///< Which edge of the element is coupled to the acoustic
                    ///< element
  specfem::kokkos::DeviceView1d<specfem::enums::coupling::edge::type>
      acoustic_edge; ///< Which edge of the element is coupled to the elastic
                     ///< element
  specfem::kokkos::HostMirror1d<specfem::enums::coupling::edge::type>
      h_elastic_edge; ///< Which edge of the element is coupled to the acoustic
                      ///< element (host mirror)
  specfem::kokkos::HostMirror1d<specfem::enums::coupling::edge::type>
      h_acoustic_edge; ///< Which edge of the element is coupled to the elastic
                       ///< element (host mirror)
  int num_interfaces;

  elastic_acoustic(
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostView1d<specfem::enums::element::type>
          h_ispec_type,
      const specfem::mesh::coupled_interfaces::elastic_acoustic
          &elastic_acoustic);
};
struct elastic_poroelastic {
  specfem::kokkos::HostMirror1d<int> h_elastic_ispec; ///< Index of the spectal
                                                      ///< element on the
                                                      ///< elastic side of the
                                                      ///< interface (host
                                                      ///< mirror)
  specfem::kokkos::HostMirror1d<int>
      h_poroelastic_ispec; ///< Index of the spectal element on the poroelastic
                           ///< side of the interface (host mirror)
  specfem::kokkos::DeviceView1d<int> elastic_ispec; ///< Index of the spectal
                                                    ///< element on the elastic
                                                    ///< side of the interface
  specfem::kokkos::DeviceView1d<int> poroelastic_ispec; ///< Index of the
                                                        ///< spectal element on
                                                        ///< the poroelastic
                                                        ///< side of the
                                                        ///< interface
  specfem::kokkos::DeviceView1d<specfem::enums::coupling::edge::type>
      elastic_edge; ///< Which edge of the element is coupled to the poroelastic
                    ///< element
  specfem::kokkos::DeviceView1d<specfem::enums::coupling::edge::type>
      poroelastic_edge; ///< Which edge of the element is coupled to the elastic
                        ///< element
  specfem::kokkos::HostMirror1d<specfem::enums::coupling::edge::type>
      h_elastic_edge; ///< Which edge of the element is coupled to the
                      ///< poroelastic element (host mirror)
  specfem::kokkos::HostMirror1d<specfem::enums::coupling::edge::type>
      h_poroelastic_edge; ///< Which edge of the element is coupled to the
                          ///< elastic element (host mirror)
  int num_interfaces;

  elastic_poroelastic(
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostView1d<specfem::enums::element::type>
          h_ispec_type,
      const specfem::mesh::coupled_interfaces::elastic_poroelastic
          &elastic_poroelastic);
};
struct acoustic_poroelastic {
  specfem::kokkos::HostMirror1d<int> h_acoustic_ispec; ///< Index of the spectal
                                                       ///< element on the
                                                       ///< acoustic side of the
                                                       ///< interface (host
                                                       ///< mirror)
  specfem::kokkos::HostMirror1d<int>
      h_poroelastic_ispec; ///< Index of the spectal element on the poroelastic
                           ///< side of the interface (host mirror)
  specfem::kokkos::DeviceView1d<int> acoustic_ispec; ///< Index of the spectal
                                                     ///< element on the
                                                     ///< acoustic side of the
                                                     ///< interface
  specfem::kokkos::DeviceView1d<int> poroelastic_ispec; ///< Index of the
                                                        ///< spectal element on
                                                        ///< the poroelastic
                                                        ///< side of the
                                                        ///< interface
  specfem::kokkos::DeviceView1d<specfem::enums::coupling::edge::type>
      acoustic_edge; ///< Which edge of the element is coupled to the
                     ///< poroelastic element
  specfem::kokkos::DeviceView1d<specfem::enums::coupling::edge::type>
      poroelastic_edge; ///< Which edge of the element is coupled to the
                        ///< acoustic element
  specfem::kokkos::HostMirror1d<specfem::enums::coupling::edge::type>
      h_acoustic_edge; ///< Which edge of the element is coupled to the
                       ///< poroelastic element (host mirror)
  specfem::kokkos::HostMirror1d<specfem::enums::coupling::edge::type>
      h_poroelastic_edge; ///< Which edge of the element is coupled to the
                          ///< acoustic element (host mirror)
  int num_interfaces;

  acoustic_poroelastic(
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostView1d<specfem::enums::element::type>
          h_ispec_type,
      const specfem::mesh::coupled_interfaces::acoustic_poroelastic
          &acoustic_poroelastic);
};

struct coupled_interfaces {
public:
  specfem::compute::coupled_interfaces::elastic_acoustic elastic_acoustic;
  specfem::compute::coupled_interfaces::elastic_poroelastic elastic_poroelastic;
  specfem::compute::coupled_interfaces::acoustic_poroelastic
      acoustic_poroelastic;

  coupled_interfaces(
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostView1d<specfem::enums::element::type>
          h_ispec_type,
      const specfem::mesh::coupled_interfaces::coupled_interfaces
          &coupled_interfaces);
};

namespace iterator {

int get_npoints(const specfem::enums::coupling::edge::type &edge,
                const int ngllx, const int ngllz);

void get_points_along_the_edges(
    const int &ipoint, const specfem::enums::coupling::edge::type &edge,
    const int &ngllx, const int &ngllz, int &i, int &j);
} // namespace iterator
} // namespace coupled_interfaces
} // namespace compute
} // namespace specfem

#endif
