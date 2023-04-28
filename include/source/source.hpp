#ifndef _SOURCE_HPP
#define _SOURCE_HPP

#include "enums.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utils.h"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace sources {

/**
 * @brief Base source class
 *
 */
class source {

public:
  /**
   * @brief Default source constructor
   *
   */
  source(){};
  /**
   * @brief Locate source within the mesh
   *
   * Given the global cartesian coordinates of a source, locate the spectral
   * element and xi, gamma value of the source
   *
   * @param h_ibool Global number for every quadrature point
   * @param coord (x, z) for every distinct control node
   * @param xigll Quadrature points in x-dimension
   * @param zigll Quadrature points in z-dimension
   * @param nproc Number of processors in the simulation
   * @param coorg Value of every spectral element control nodes
   * @param knods Global control element number for every control node
   * @param npgeo Total number of distinct control nodes
   * @param ispec_type material type for every spectral element
   * @param mpi Pointer to specfem MPI object
   */
  virtual void locate(
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostMirror1d<type_real> xigll,
      const specfem::kokkos::HostMirror1d<type_real> zigll, const int nproc,
      const specfem::kokkos::HostView2d<type_real> coorg,
      const specfem::kokkos::HostView2d<int> knods, const int npgeo,
      const specfem::kokkos::HostMirror1d<specfem::elements::type> ispec_type,
      const specfem::MPI::MPI *mpi){};
  /**
   * @brief Precompute and store lagrangian values used to compute integrals for
   * sources
   *
   * @param quadx Quadrature object in x-dimension
   * @param quadz Quadrature object in z-dimension
   * @param source_array view to store the source array
   */
  virtual void
  compute_source_array(const specfem::quadrature::quadrature *quadx,
                       const specfem::quadrature::quadrature *quadz,
                       specfem::kokkos::HostView3d<type_real> source_array){};
  /**
   * @brief Check if the source is within the domain
   *
   * @param xmin minimum x-coordinate on my processor
   * @param xmax maximum x-coordinate on my processor
   * @param zmin minimum z-coordinate on my processor
   * @param zmax maximum z-coordinate on my processor
   * @param mpi Pointer to specfem MPI object
   */
  virtual void check_locations(const type_real xmin, const type_real xmax,
                               const type_real zmin, const type_real zmax,
                               const specfem::MPI::MPI *mpi);
  /**
   * @brief Get the processor on which this source lies
   *
   * @return int value of processor
   */
  virtual int get_islice() const { return 0; }
  /**
   * @brief Get the element inside which this source lies
   *
   * @return int value of element
   */
  virtual int get_ispec() const { return 0; }
  /**
   * @brief Get the x coordinate of the source
   *
   * @return type_real x-coordinate
   */
  virtual type_real get_x() const { return 0.0; }
  /**
   * @brief Get the z coordinate of the source
   *
   * @return type_real z-coordinate
   */
  virtual type_real get_z() const { return 0.0; }
  /**
   * @brief Get the \f$ \xi \f$ value of the source within the element
   *
   * @return type_real \f$ \xi \f$ value
   */
  virtual type_real get_xi() const { return 0.0; }
  /**
   * @brief Get the \f$ \gamma \f$ value of the source within the element
   *
   * @return type_real \f$ \gamma \f$ value
   */
  virtual type_real get_gamma() const { return 0.0; }
  /**
   * @brief Get the value of t0 from the specfem::stf::stf object
   *
   * @return value of t0
   */
  KOKKOS_IMPL_HOST_FUNCTION
  virtual type_real get_t0() const { return 0.0; }
  /**
   * @brief Update the value of tshift for specfem::stf::stf object
   *
   * @return new value of tshift
   */
  virtual void update_tshift(type_real tshift){};
  /**
   * @brief User output
   *
   */
  virtual void print(std::ostream &out) const;
  /**
   * @brief User output
   *
   */
  virtual std::string print() const { return ""; };
  /**
   * @brief Get the device pointer to stf object
   *
   * @return specfem::forcing_function::stf*
   */
  virtual specfem::forcing_function::stf *get_stf() const {
    return new specfem::forcing_function::stf();
  }
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::sources::source &source);

} // namespace sources

} // namespace specfem
#endif
