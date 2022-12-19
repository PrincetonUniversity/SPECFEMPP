#ifndef SOURCES_H
#define SOURCES_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/source_time_function.h"
#include "../include/specfem_mpi.h"
#include "../include/timescheme.h"
#include "../include/utils.h"

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
   * @param ibool Global number for every quadrature point
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
  virtual void locate(const specfem::HostView3d<int> ibool,
                      const specfem::HostView2d<type_real> coord,
                      const specfem::HostMirror1d<type_real> xigll,
                      const specfem::HostMirror1d<type_real> zigll,
                      const int nproc,
                      const specfem::HostView2d<type_real> coorg,
                      const specfem::HostView2d<int> knods, const int npgeo,
                      const specfem::HostView1d<element_type> ispec_type,
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
  compute_source_array(specfem::quadrature::quadrature &quadx,
                       specfem::quadrature::quadrature &quadz,
                       specfem::HostView3d<type_real> source_array){};
  /**
   * @brief Compute source time function
   *
   */
  virtual void
  compute_stf(specfem::HostView1d<type_real, Kokkos::LayoutStride> stf_array,
              specfem::TimeScheme::TimeScheme *it){};
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
  virtual type_real get_t0() const { return 0.0; }
  virtual void update_tshift(type_real tshift){};
  virtual void print(std::ostream &out) const;
};

/**
 * @brief Collocated force source
 *
 */
class force : public source {

public:
  /**
   * @brief Construct a new collocated force object
   *
   * @param x x-coordinate of the source
   * @param z z-coordinate of the source
   * @param angle angle of the source
   * @param wave Type of simulation P-SV or SH wave simulation
   */
  force(type_real x, type_real z, type_real angle, type_real tshift,
        type_real f0, type_real factor, std::string forcing_type,
        wave_type wave);
  /**
   * @brief Construct a new collocated force object
   *
   * @param force_source a force_struct data holder read from source file
   * written in .yml format
   * @param wave Type of simulation P-SV or SH wave simulation
   */
  force(specfem::utilities::force_source &force_source, wave_type wave);
  /**
   * @brief Locate source within the mesh
   *
   * Given the global cartesian coordinates of a source, locate the spectral
   * element and xi, gamma value of the source
   *
   * @param ibool Global number for every quadrature point
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
  void locate(const specfem::HostView3d<int> ibool,
              const specfem::HostView2d<type_real> coord,
              const specfem::HostMirror1d<type_real> xigll,
              const specfem::HostMirror1d<type_real> zigll, const int nproc,
              const specfem::HostView2d<type_real> coorg,
              const specfem::HostView2d<int> knods, const int npgeo,
              const specfem::HostView1d<element_type> ispec_type,
              const specfem::MPI::MPI *mpi) override;
  /**
   * @brief Precompute and store lagrangian values used to compute integrals for
   * sources
   *
   * @param quadx Quadrature object in x-dimension
   * @param quadz Quadrature object in z-dimension
   * @param source_array view to store the source array
   */
  void
  compute_source_array(specfem::quadrature::quadrature &quadx,
                       specfem::quadrature::quadrature &quadz,
                       specfem::HostView3d<type_real> source_array) override;
  void
  compute_stf(specfem::HostView1d<type_real, Kokkos::LayoutStride> stf_array,
              specfem::TimeScheme::TimeScheme *it) override;
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
                       const specfem::MPI::MPI *mpi);
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
  ~force() { delete this->forcing_function; }
  type_real get_t0() const override { return this->forcing_function->get_t0(); }
  void update_tshift(type_real tshift) override {
    this->forcing_function->update_tshift(tshift);
  }
  void print(std::ostream &out) const override;

private:
  type_real xi;         ///< \f$ \xi \f$ value of source inside element
  type_real gamma;      ///< \f$ \gamma \f$ value of source inside element
  type_real x;          ///< x coordinate of source
  type_real z;          ///< z coordinate of source
  type_real angle;      ///< angle of the source
  int ispec;            ///< ispec element number where source is located
  int islice;           ///< MPI slice (rank) where the source is located
  element_type el_type; ///< type of the element inside which this source lies
  wave_type wave;       ///< SH or P-SV wave
  specfem::forcing_function::stf *forcing_function;
};

/**
 * @brief Moment-tensor source
 *
 */
class moment_tensor : public source {

public:
  /**
   * @brief Construct a new moment tensor force object
   *
   * @param x x-coordinate of moment tensor source
   * @param z z-coordinate of moment tensor source
   * @param Mxx Mxx for the source
   * @param Mxz Mxz for the source
   * @param Mzz Mzz for the source
   */
  moment_tensor(type_real x, type_real z, type_real Mxx, type_real Mxz,
                type_real Mzz, type_real tshift, type_real f0, type_real factor,
                std::string forcing_type);
  /**
   * @brief Construct a new moment tensor force object
   *
   * @param moment_tensor a moment_tensor data holder read from source file
   * written in .yml format
   */
  moment_tensor(specfem::utilities::moment_tensor &moment_tensor);
  /**
   * @brief Locate source within the mesh
   *
   * Given the global cartesian coordinates of a source, locate the spectral
   * element and xi, gamma value of the source
   *
   * @param ibool Global number for every quadrature point
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
  void locate(const specfem::HostView3d<int> ibool,
              const specfem::HostView2d<type_real> coord,
              const specfem::HostMirror1d<type_real> xigll,
              const specfem::HostMirror1d<type_real> zigll, const int nproc,
              const specfem::HostView2d<type_real> coorg,
              const specfem::HostView2d<int> knods, const int npgeo,
              const specfem::HostView1d<element_type> ispec_type,
              const specfem::MPI::MPI *mpi) override;
  /**
   * @brief Precompute and store lagrangian values used to compute integrals for
   * sources
   *
   * @param quadx Quadrature object in x-dimension
   * @param quadz Quadrature object in z-dimension
   * @param source_array view to store the source array
   */
  void
  compute_source_array(specfem::quadrature::quadrature &quadx,
                       specfem::quadrature::quadrature &quadz,
                       specfem::HostView3d<type_real> source_array) override;
  void
  compute_stf(specfem::HostView1d<type_real, Kokkos::LayoutStride> stf_array,
              specfem::TimeScheme::TimeScheme *it) override;
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
  type_real get_t0() const override { return this->forcing_function->get_t0(); }
  void update_tshift(type_real tshift) {
    this->forcing_function->update_tshift(tshift);
  }
  friend std::ostream &operator<<(std::ostream &out, const moment_tensor *h);
  void print(std::ostream &out) const override;
  ~moment_tensor() { delete this->forcing_function; }

private:
  type_real xi;    ///< \f$ \xi \f$ value of source inside element
  type_real gamma; ///< \f$ \gamma \f$ value of source inside element
  type_real x;     ///< x coordinate of source
  type_real z;     ///< z coordinate of source
  type_real Mxx;   ///< Mxx for the source
  type_real Mxz;   ///< Mxz for the source
  type_real Mzz;   ///< Mzz for the source
  int ispec;       ///< ispec element number where source is located
  int islice;      ///< MPI slice (rank) where the source is located
  specfem::HostView2d<type_real> s_coorg; ///< control nodes subviewed at the
                                          ///< element where this source is
                                          ///< located

  specfem::forcing_function::stf *forcing_function;
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::sources::source &source);

} // namespace sources

} // namespace specfem
#endif
