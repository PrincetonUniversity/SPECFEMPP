#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"

namespace specfem {
namespace recievers {

class reciever {

public:
  reciever(type_real x, type_real z) : x(x), z(z){};
  void locate(const specfem::HostView3d<int> ibool,
              const specfem::HostView2d<type_real> coord,
              const specfem::HostMirror1d<type_real> xigll,
              const specfem::HostMirror1d<type_real> zigll, const int nproc,
              const specfem::HostView3d<type_real> coorg,
              const specfem::HostView2d<int> knods, const int npgeo,
              const specfem::HostView1d<element_type> ispec_type,
              const specfem::MPI *mpi){};
  void compute_reciever_array(specfem::quadrature &quadx,
                              specfem::quadrature &quadz,
                              specfem::HostView3d<type_real> reciever_array){};
  void check_locations(const type_real xmin, const type_real xmax,
                       const type_real zmin, const type_real zmax,
                       specfem::MPI *mpi);

private:
  type_real xi;         ///< f$ \xi f$ value of source inside element
  type_real gamma;      ///< f$ \gamma f$ value of source inside element
  type_real x;          ///< x coordinate of source
  type_real z;          ///< z coordinate of source
  int ispec;            ///< ispec element number where source is located
  int islice;           ///< MPI slice (rank) where the source is located
  element_type el_type; ///< type of the element inside which this reciever lies
}
} // namespace recievers

} // namespace specfem
