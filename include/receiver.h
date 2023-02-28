#ifndef RECEIVER_H
#define RECEIVER_H

#include "../include/config.h"
#include "../include/constants.h"
#include "../include/enums.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/specfem_mpi.h"
#include <cmath>

namespace specfem {
namespace receivers {

class receiver {

public:
  receiver(const type_real x, const type_real z, const type_real angle)
      : x(x), z(z), angle(angle){};
  void
  locate(const specfem::kokkos::HostView3d<int> ibool,
         const specfem::kokkos::HostView2d<type_real> coord,
         const specfem::kokkos::HostMirror1d<type_real> xigll,
         const specfem::kokkos::HostMirror1d<type_real> zigll, const int nproc,
         const specfem::kokkos::HostView2d<type_real> coorg,
         const specfem::kokkos::HostView2d<int> knods, const int npgeo,
         const specfem::kokkos::HostView1d<specfem::elements::type> ispec_type,
         const specfem::MPI::MPI *mpi);
  void
  compute_receiver_array(const specfem::quadrature::quadrature &quadx,
                         const specfem::quadrature::quadrature &quadz,
                         specfem::kokkos::HostView3d<type_real> receiver_array);
  void check_locations(const type_real xmin, const type_real xmax,
                       const type_real zmin, const type_real zmax,
                       const specfem::MPI::MPI *mpi);
  int get_islice() { return this->islice; };
  int get_ispec() { return this->ispec; };
  type_real get_sine() {
    return std::sin(Kokkos::Experimental::pi_v<type_real> / 180 * this->angle);
  }
  type_real get_cosine() {
    return std::cos(Kokkos::Experimental::pi_v<type_real> / 180 * this->angle);
  }

private:
  type_real xi;    ///< f$ \xi f$ value of source inside element
  type_real gamma; ///< f$ \gamma f$ value of source inside element
  type_real x;     ///< x coordinate of source
  type_real z;     ///< z coordinate of source
  int ispec;       ///< ispec element number where source is located
  int islice;      ///< MPI slice (rank) where the source is located
  specfem::elements::type el_type; ///< type of the element inside which this
                                   ///< receiver lies
  type_real angle;                 ///< Angle to rotate components at receivers
};
} // namespace receivers

} // namespace specfem

#endif
