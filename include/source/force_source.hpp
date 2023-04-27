#ifndef _FORCE_SOURCE_HPP
#define _FORCE_SOURCE_HPP

#include "enums.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_mpi.h"
#include "specfem_setup.hpp"
#include "utils.h"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace sources {
/**
 * @brief Collocated force source
 *
 */
class force : public source {

public:
  /**
   * @brief Construct a new collocated force object
   *
   * @param force_source A YAML node defining force source
   * @param dt Time increment in the simulation. Used to calculate dominant
   * frequecy of Dirac source.
   */
  force(YAML::Node &Node, const type_real dt);
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
  void locate(
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostMirror1d<type_real> xigll,
      const specfem::kokkos::HostMirror1d<type_real> zigll, const int nproc,
      const specfem::kokkos::HostView2d<type_real> coorg,
      const specfem::kokkos::HostView2d<int> knods, const int npgeo,
      const specfem::kokkos::HostMirror1d<specfem::elements::type> ispec_type,
      const specfem::MPI::MPI *mpi) override;
  /**
   * @brief Precompute and store lagrangian values used to compute integrals for
   * sources
   *
   * @param quadx Quadrature object in x-dimension
   * @param quadz Quadrature object in z-dimension
   * @param source_array view to store the source array
   */
  void compute_source_array(
      const specfem::quadrature::quadrature *quadx,
      const specfem::quadrature::quadrature *quadz,
      specfem::kokkos::HostView3d<type_real> source_array) override;
  /**
   * @brief Check if the source is within the domain
   *
   * @param xmin minimum x-coordinate on my processor
   * @param xmax maximum x-coordinate on my processor
   * @param zmin minimum z-coordinate on my processor
   * @param zmax maximum z-coordinate on my processor
   * @param mpi Pointer to specfem MPI object
   */
  void check_locations(const type_real xmin, const type_real xmax,
                       const type_real zmin, const type_real zmax,
                       const specfem::MPI::MPI *mpi) override;
  /**
   * @brief Get the processor on which this source lies
   *
   * @return int value of processor
   */
  int get_islice() const override { return this->islice; }
  /**
   * @brief Get the element inside which this source lies
   *
   * @return int value of element
   */
  int get_ispec() const override { return this->ispec; }
  /**
   * @brief Get the x coordinate of the source
   *
   * @return type_real x-coordinate
   */
  type_real get_x() const override { return x; }
  /**
   * @brief Get the z coordinate of the source
   *
   * @return type_real z-coordinate
   */
  type_real get_z() const override { return z; }
  /**
   * @brief Get the \f$ \xi \f$ value of the source within the element
   *
   * @return type_real \f$ \xi \f$ value
   */
  type_real get_xi() const override { return xi; }
  /**
   * @brief Get the \f$ \gamma \f$ value of the source within the element
   *
   * @return type_real \f$ \gamma \f$ value
   */
  type_real get_gamma() const override { return gamma; }
  /**
   * @brief Destroy the force object
   *
   */
  ~force() {
    Kokkos::kokkos_free<specfem::kokkos::DevMemSpace>(this->forcing_function);
  }
  /**
   * @brief Get the value of t0 from the specfem::stf::stf object
   *
   * @return value of t0
   */
  KOKKOS_IMPL_HOST_FUNCTION
  type_real get_t0() const override;
  /**
   * @brief Update the value of tshift for specfem::stf::stf object
   *
   * @return new value of tshift
   */
  void update_tshift(type_real tshift) override;
  /**
   * @brief User output
   *
   */
  void print(std::ostream &out) const override;
  /**
   * @brief User output
   *
   */
  std::string print() const override;
  /**
   * @brief Get the device pointer to stf object
   *
   * @return specfem::forcing_function::stf*
   */
  specfem::forcing_function::stf *get_stf() const override {
    return forcing_function;
  }

private:
  type_real xi;    ///< \f$ \xi \f$ value of source inside element
  type_real gamma; ///< \f$ \gamma \f$ value of source inside element
  type_real x;     ///< x coordinate of source
  type_real z;     ///< z coordinate of source
  type_real angle; ///< angle of the source
  int ispec;       ///< ispec element number where source is located
  int islice;      ///< MPI slice (rank) where the source is located
  specfem::elements::type el_type; ///< type of the element inside which this
                                   ///< source lies
  specfem::forcing_function::stf *forcing_function =
      NULL; ///< Pointer to source time function store on the device
};
} // namespace sources
} // namespace specfem

#endif
