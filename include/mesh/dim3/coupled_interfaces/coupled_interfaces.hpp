#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"

namespace specfem {
namespace mesh {

template <specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2>
struct coupled_interface {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

  int nelements;                    ///< Number of elements on the boundary
  int ngllsquare;                   ///< Number of GLL points squared
  int num_coupling_interface_faces; ///< Number of coupling interface faces
  int nspec2D_xmin, nspec2D_xmax;   /// Number of elements on the boundaries
  int nspec2D_ymin, nspec2D_ymax;   /// Number of elements on the boundaries
  int NSPEC2D_BOTTOM, NSPEC2D_TOP;  /// Number of elements on the boundaries

  Kokkos::View<int *, Kokkos::HostSpace> ispec; ///< Spectral element index for
                                                ///< elements on the boundary
  Kokkos::View<int ***, Kokkos::HostSpace> ijk; ///< Which edge of the element
                                                ///< is on the boundary
  Kokkos::View<type_real **, Kokkos::HostSpace> jacobian2Dw; ///< Jacobian of
                                                             ///< the 2D
  Kokkos::View<type_real ***, Kokkos::HostSpace> normal; ///< Jacobian of the 2D

  // Default constructor
  coupled_interface(){};

  coupled_interface(const int num_coupling_interface_faces,
                    const int ngllsquare)
      : nelements(num_coupling_interface_faces), ngllsquare(ngllsquare),
        num_coupling_interface_faces(num_coupling_interface_faces) {

    ispec = Kokkos::View<int *, Kokkos::HostSpace>("ispec", nelements);
    ijk = Kokkos::View<int ***, Kokkos::HostSpace>("ijk", nelements, 3,
                                                   ngllsquare);
    jacobian2Dw = Kokkos::View<type_real **, Kokkos::HostSpace>(
        "jacobian2Dw", nelements, ngllsquare);
    normal = Kokkos::View<type_real ***, Kokkos::HostSpace>("normal", nelements,
                                                            3, ngllsquare);
  };

  void print() const;
};

template <> struct coupled_interfaces<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

  bool acoustic_elastic = false; ///< Flag for acoustic elastic interfaces
  bool acoustic_poroelastic =
      false; ///< Flag for acoustic poroelastic interfaces
  bool elastic_poroelastic = false; ///< Flag for elastic poroelastic interfaces

  int num_coupling_ac_el_faces; ///< Number of coupling acoustic elastic faces
  int num_coupling_ac_po_faces; ///< Number of coupling acoustic poroelastic
                                ///< faces
  int num_coupling_el_po_faces; ///< Number of coupling elastic poroelastic
                                ///< faces
  int ngllsquare;               ///< Number of GLL points squared

  // Create coupled placeholders.
  coupled_interface<specfem::element::medium_tag::acoustic,
                    specfem::element::medium_tag::elastic>
      acoustic_elastic_interface;
  coupled_interface<specfem::element::medium_tag::acoustic,
                    specfem::element::medium_tag::poroelastic>
      acoustic_poroelastic_interface;
  coupled_interface<specfem::element::medium_tag::elastic,
                    specfem::element::medium_tag::poroelastic>
      elastic_poroelastic_interface;
  coupled_interface<specfem::element::medium_tag::poroelastic,
                    specfem::element::medium_tag::elastic>
      poroelastic_elastic_interface;

  // Default constructor
  coupled_interfaces(){};

  coupled_interfaces(const int num_coupling_ac_el_faces,
                     const int num_coupling_ac_po_faces,
                     const int num_coupling_el_po_faces, const int ngllsquare)
      : num_coupling_ac_el_faces(num_coupling_ac_el_faces),
        num_coupling_ac_po_faces(num_coupling_ac_po_faces),
        num_coupling_el_po_faces(num_coupling_el_po_faces),
        ngllsquare(ngllsquare) {

    if (num_coupling_ac_el_faces > 0) {
      acoustic_elastic_interface =
          coupled_interface<specfem::element::medium_tag::acoustic,
                            specfem::element::medium_tag::elastic>(
              num_coupling_ac_el_faces, ngllsquare);
      acoustic_elastic = true;
    }
    if (num_coupling_ac_po_faces > 0) {
      acoustic_poroelastic_interface =
          coupled_interface<specfem::element::medium_tag::acoustic,
                            specfem::element::medium_tag::poroelastic>(
              num_coupling_ac_po_faces, ngllsquare);
      acoustic_poroelastic = true;
    }
    if (num_coupling_el_po_faces > 0) {
      elastic_poroelastic_interface =
          coupled_interface<specfem::element::medium_tag::elastic,
                            specfem::element::medium_tag::poroelastic>(
              num_coupling_el_po_faces, ngllsquare);

      poroelastic_elastic_interface =
          coupled_interface<specfem::element::medium_tag::poroelastic,
                            specfem::element::medium_tag::elastic>(
              num_coupling_el_po_faces, ngllsquare);

      elastic_poroelastic = true;
    }
  }

  void print() const;
};

} // namespace mesh
} // namespace specfem
