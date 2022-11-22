#ifndef SOURCES_H
#define SOURCES_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/specfem_mpi.h"

namespace specfem {
namespace sources {

class source {

public:
  virtual void locate(const specfem::HostView3d<int> ibool,
                      const specfem::HostView2d<type_real> coord,
                      const specfem::HostMirror1d<type_real> xigll,
                      const specfem::HostMirror1d<type_real> zigll,
                      const int nproc,
                      const specfem::HostView3d<type_real> coorg,
                      const specfem::HostView2d<int> knods, const int npgeo,
                      const specfem::HostView1d<element_type> ispec_type,
                      const specfem::MPI::MPI *mpi){};
  virtual void
  compute_source_array(quadrature::quadrature &quadx,
                       quadrature::quadrature &quadz,
                       specfem::HostView3d<type_real> source_array){};
  virtual void compute_stf(){};
  void check_locations(const type_real xmin, const type_real xmax,
                       const type_real zmin, const type_real zmax,
                       const specfem::MPI::MPI *mpi);
  virtual int get_islice();
  virtual int get_ispec();
};

class force : public source {

public:
  force(type_real x, type_real z, type_real angle, wave_type wave)
      : x(x), z(z), angle(angle), wave(wave){};
  force(specfem::utilities::force_source &force_source, wave_type wave)
      : x(force_source.x), z(force_source.z), angle(force_source.angle),
        wave(wave){};
  void locate(const specfem::HostView3d<int> ibool,
              const specfem::HostView2d<type_real> coord,
              const specfem::HostMirror1d<type_real> xigll,
              const specfem::HostMirror1d<type_real> zigll, const int nproc,
              const specfem::HostView3d<type_real> coorg,
              const specfem::HostView2d<int> knods, const int npgeo,
              const specfem::HostView1d<element_type> ispec_type,
              const specfem::MPI::MPI *mpi) override;
  void
  compute_source_array(quadrature::quadrature &quadx,
                       quadrature::quadrature &quadz,
                       specfem::HostView3d<type_real> source_array) override;
  void compute_stf() override;
  void check_locations(const type_real xmin, const type_real xmax,
                       const type_real zmin, const type_real zmax,
                       const specfem::MPI::MPI *mpi);
  int get_islice() override { return this->islice; }
  int get_ispec() override { return this->ispec; }

private:
  type_real xi;         ///< f$ \xi f$ value of source inside element
  type_real gamma;      ///< f$ \gamma f$ value of source inside element
  type_real x;          ///< x coordinate of source
  type_real z;          ///< z coordinate of source
  type_real angle;      ///< angle of the source
  int ispec;            ///< ispec element number where source is located
  int islice;           ///< MPI slice (rank) where the source is located
  element_type el_type; ///< type of the element inside which this source lies
  wave_type wave;       ///< SH or P-SV wave
};

class moment_tensor : public source {

public:
  moment_tensor(type_real x, type_real z, type_real Mxx, type_real Mxz,
                type_real Mzz)
      : x(x), z(z), Mxx(Mxx), Mxz(Mxz), Mzz(Mzz){};
  moment_tensor(specfem::utilities::moment_tensor &moment_tensor)
      : x(moment_tensor.x), z(moment_tensor.z), Mxx(moment_tensor.Mxx),
        Mxz(moment_tensor.Mxz), Mzz(moment_tensor.Mzz){};
  void locate(const specfem::HostView3d<int> ibool,
              const specfem::HostView2d<type_real> coord,
              const specfem::HostMirror1d<type_real> xigll,
              const specfem::HostMirror1d<type_real> zigll, const int nproc,
              const specfem::HostView3d<type_real> coorg,
              const specfem::HostView2d<int> knods, const int npgeo,
              const specfem::HostView1d<element_type> ispec_type,
              const specfem::MPI::MPI *mpi) override;
  void
  compute_source_array(quadrature::quadrature &quadx,
                       quadrature::quadrature &quadz,
                       specfem::HostView3d<type_real> source_array) override;
  void compute_stf() override;
  int get_islice() override { return this->islice; }
  int get_ispec() override { return this->ispec; }

private:
  type_real xi;    ///< f$ \xi f$ value of source inside element
  type_real gamma; ///< f$ \gamma f$ value of source inside element
  type_real x;     ///< x coordinate of source
  type_real z;     ///< z coordinate of source
  type_real Mxx;   ///< Mxx for the source
  type_real Mxz;   ///< Mxz for the source
  type_real Mzz;   ///< Mzz for the source
  int ispec;       ///< ispec element number where source is located
  int islice;      ///< MPI slice (rank) where the source is located
};
} // namespace sources

} // namespace specfem

#endif
