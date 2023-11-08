#ifndef _COMPUTE_COUPLED_INTERFACES_HPP
#define _COMPUTE_COUPLED_INTERFACES_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "mesh/coupled_interfaces/coupled_interfaces.hpp"

namespace specfem {
namespace compute {
namespace coupled_interfaces {
/**
 * @brief Elastic-Acoustic interface information
 *
 */
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
  int num_interfaces;  ///< Total number of edges coupled between elastic and
                       ///< acoustic elements

  /**
   * @brief Construct a new elastic acoustic interface
   *
   * @param h_ibool Global Index for every quadrature point in the mesh
   * @param coord coordinates of every quadrature point in the mesh
   * @param h_ispec_type Element type (acoustic/elastic) for every element in
   * the mesh
   * @param elastic_acoustic Interface information for every elastic-acoustic
   * interface read from mesher
   */
  elastic_acoustic(
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostView1d<specfem::enums::element::type>
          h_ispec_type,
      const specfem::mesh::coupled_interfaces::elastic_acoustic
          &elastic_acoustic);
};
/**
 * @brief Elastic-Poroelastic interface information
 *
 */
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
  int num_interfaces;     ///< Total number of edges coupled between elastic and
                          ///< poroelastic elements

  /**
   * @brief Contruct a new elastic poroelastic interface
   *
   * @param h_ibool Index for every quadrature point in the mesh
   * @param coord Coordinates of every quadrature point in the mesh
   * @param h_ispec_type Element type (elastic/poroelastic) for every element in
   * the mesh
   * @param elastic_poroelastic Interface information for every
   * elastic-poroelastic interface read from mesher
   */
  elastic_poroelastic(
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostView1d<specfem::enums::element::type>
          h_ispec_type,
      const specfem::mesh::coupled_interfaces::elastic_poroelastic
          &elastic_poroelastic);
};
/**
 * @brief Acoustic-Poroelastic interface information
 *
 */
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
  int num_interfaces; ///< Total number of edges coupled between acoustic and
                      ///< poroelastic elements

  /**
   * @brief Construct a new acoustic poroelastic interface
   *
   * @param h_ibool Index for every quadrature point in the mesh
   * @param coord Coordinates of every quadrature point in the mesh
   * @param h_ispec_type Element type (acoustic/poroelastic) for every element
   * in the mesh
   * @param acoustic_poroelastic Interface information for every
   * acoustic-poroelastic interface read from mesher
   */
  acoustic_poroelastic(
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostView1d<specfem::enums::element::type>
          h_ispec_type,
      const specfem::mesh::coupled_interfaces::acoustic_poroelastic
          &acoustic_poroelastic);
};

/**
 * @brief Struct used to store all interface information at every interfacial
 * quadrature point within the mesh
 *
 */
struct coupled_interfaces {
public:
  specfem::compute::coupled_interfaces::elastic_acoustic
      elastic_acoustic; ///< Elastic-Acoustic interface information
  specfem::compute::coupled_interfaces::elastic_poroelastic
      elastic_poroelastic; ///< Elastic-Poroelastic interface information
  specfem::compute::coupled_interfaces::acoustic_poroelastic
      acoustic_poroelastic; ///< Acoustic-Poroelastic interface information

  /**
   * @brief Construct a new coupled interfaces object
   *
   * @param h_ibool Index for every quadrature point in the mesh
   * @param coord  Coordinates of every quadrature point in the mesh
   * @param h_ispec_type Element type (acoustic/elastic/poroelastic) for every
   * element in the mesh
   * @param coupled_interfaces Interface information for every interface read
   * from mesher
   */
  coupled_interfaces(
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostView1d<specfem::enums::element::type>
          h_ispec_type,
      const specfem::mesh::coupled_interfaces::coupled_interfaces
          &coupled_interfaces);
};

/**
 * @namespace Helper functions to access the edges at an interface
 *
 */
namespace access {

/**
 * @brief Compute the number of points at an edge of an element
 *
 * @param edge Orientation of the edge
 * @param ngllx Number of GLL points in the x-direction
 * @param ngllz Number of GLL points in the z-direction
 */
KOKKOS_FUNCTION
int npoints(const specfem::enums::coupling::edge::type &edge, const int ngllx,
            const int ngllz);

/**
 * @brief Get the index of quadrature point of self domain (ix, iz) at an edge
 * of an element given the edge orientation and index of the quadrature point on
 * that edge
 *
 * @param ipoint Index of the quadrature point on the edge
 * @param edge Orientation of the edge
 * @param ngllx Number of GLL points in the x-direction
 * @param ngllz Number of GLL points in the z-direction
 * @param i ix index of the quadrature point
 * @param j iz index of the quadrature point
 */
KOKKOS_FUNCTION
void self_iterator(const int &ipoint,
                   const specfem::enums::coupling::edge::type &edge,
                   const int ngllx, const int ngllz, int &i, int &j);

/**
 * @brief Get the index of quadrature point of coupled domain (ix, iz) at an
 * edge of an element given the edge orientation and index of the quadrature
 * point on that edge
 *
 * @param ipoint Index of the quadrature point on the edge
 * @param edge Orientation of the edge
 * @param ngllx Number of GLL points in the x-direction
 * @param ngllz Number of GLL points in the z-direction
 * @param i ix index of the quadrature point
 * @param j iz index of the quadrature point
 */
KOKKOS_FUNCTION
void coupled_iterator(const int &ipoint,
                      const specfem::enums::coupling::edge::type &edge,
                      const int ngllx, const int ngllz, int &i, int &j);

} // namespace access
} // namespace coupled_interfaces
} // namespace compute
} // namespace specfem

#endif
