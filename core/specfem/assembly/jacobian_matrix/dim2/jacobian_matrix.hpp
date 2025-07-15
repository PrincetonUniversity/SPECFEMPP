#pragma once

#include "domain_view.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

template <>
struct jacobian_matrix<specfem::dimension::type::dim2>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::jacobian_matrix,
          specfem::dimension::type::dim2> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::jacobian_matrix,
      specfem::dimension::type::dim2>; ///< Base type of the point partial
                                       ///< derivatives
  using view_type = typename base_type::scalar_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  ///@}

  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z direction
  int ngllx; ///< Number of quadrature points in x direction

  view_type xix;                    ///< @xix
  view_type::HostMirror h_xix;      ///< Host mirror of @xix
  view_type xiz;                    ///< @xiz
  view_type::HostMirror h_xiz;      ///< Host mirror of @xiz
  view_type gammax;                 ///< @gammax
  view_type::HostMirror h_gammax;   ///< Host mirror of @gammax
  view_type gammaz;                 ///< @gammaz
  view_type::HostMirror h_gammaz;   ///< Host mirror of @gammaz
  view_type jacobian;               ///< Jacobian
  view_type::HostMirror h_jacobian; ///< Host mirror of Jacobian

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  jacobian_matrix() = default;

  jacobian_matrix(const int nspec, const int ngllz, const int ngllx);
  /**
   * @brief Construct a new Jacobian matrix object from mesh information
   *
   * @param mesh Mesh information
   */
  jacobian_matrix(
      const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh);
  ///@}

  void sync_views();

  /**
   * @brief Check if the Jacobian is a small value
   *
   * @return std::tuple<bool, Kokkos::View> Tuple containing a boolean
   * indicating whether a small Jacobian was found and a view containing the
   * indices of the spectral elements with small Jacobian
   */
  std::tuple<bool, Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> >
  check_small_jacobian() const;
};

/**
 * @defgroup ComputeJacobianMatrixDataAccess
 *
 * @brief Functions to load and store Jacobian matrix at a given quadrature
 * point
 *
 */

template <bool on_device, typename PointJacobianMatrixType,
          typename std::enable_if_t<PointJacobianMatrixType::simd::using_simd,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(
    const specfem::point::simd_index<PointJacobianMatrixType::dimension_tag>
        &index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &derivatives,
    PointJacobianMatrixType &jacobian_matrix) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  using simd = typename PointJacobianMatrixType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  constexpr static bool StoreJacobian = PointJacobianMatrixType::store_jacobian;

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  if constexpr (on_device) {
    Kokkos::Experimental::where(mask, jacobian_matrix.xix)
        .copy_from(&derivatives.xix[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.gammax)
        .copy_from(&derivatives.gammax[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.xiz)
        .copy_from(&derivatives.xiz[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.gammaz)
        .copy_from(&derivatives.gammaz[_index], tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, jacobian_matrix.jacobian)
          .copy_from(&derivatives.jacobian[_index], tag_type());
    }
  } else {
    Kokkos::Experimental::where(mask, jacobian_matrix.xix)
        .copy_from(&derivatives.h_xix[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.gammax)
        .copy_from(&derivatives.h_gammax[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.xiz)
        .copy_from(&derivatives.h_xiz[_index], tag_type());
    Kokkos::Experimental::where(mask, jacobian_matrix.gammaz)
        .copy_from(&derivatives.h_gammaz[_index], tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, jacobian_matrix.jacobian)
          .copy_from(&derivatives.h_jacobian[_index], tag_type());
    }
  }
}

template <bool on_device, typename PointJacobianMatrixType,
          typename std::enable_if_t<!PointJacobianMatrixType::simd::using_simd,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(
    const specfem::point::index<PointJacobianMatrixType::dimension_tag> &index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &derivatives,
    PointJacobianMatrixType &jacobian_matrix) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PointJacobianMatrixType::store_jacobian;

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  if constexpr (on_device) {
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        xix = derivatives.xix.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        gammax = derivatives.gammax.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        xiz = derivatives.xiz.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        gammaz = derivatives.gammaz.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        jacobian = derivatives.jacobian.get_base_view();

    jacobian_matrix.xix = xix(_index);
    jacobian_matrix.gammax = gammax(_index);
    jacobian_matrix.xiz = xiz(_index);
    jacobian_matrix.gammaz = gammaz(_index);
    if constexpr (StoreJacobian) {
      jacobian_matrix.jacobian = jacobian(_index);
    }
  } else {
    jacobian_matrix.xix = derivatives.h_xix[_index];
    jacobian_matrix.gammax = derivatives.h_gammax[_index];
    jacobian_matrix.xiz = derivatives.h_xiz[_index];
    jacobian_matrix.gammaz = derivatives.h_gammaz[_index];
    if constexpr (StoreJacobian) {
      jacobian_matrix.jacobian = derivatives.h_jacobian[_index];
    }
  }
}

template <typename PointJacobianMatrixType,
          typename std::enable_if_t<PointJacobianMatrixType::simd::using_simd,
                                    int> = 0>
inline void impl_store_on_host(
    const specfem::point::simd_index<PointJacobianMatrixType::dimension_tag>
        &index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &derivatives,
    const PointJacobianMatrixType &jacobian_matrix) {

  const int ispec = index.ispec;
  const int nspec = derivatives.nspec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PointJacobianMatrixType::store_jacobian;

  using simd = typename PointJacobianMatrixType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  Kokkos::Experimental::where(mask, jacobian_matrix.xix)
      .copy_to(&derivatives.h_xix[_index], tag_type());
  Kokkos::Experimental::where(mask, jacobian_matrix.gammax)
      .copy_to(&derivatives.h_gammax[_index], tag_type());
  Kokkos::Experimental::where(mask, jacobian_matrix.xiz)
      .copy_to(&derivatives.h_xiz[_index], tag_type());
  Kokkos::Experimental::where(mask, jacobian_matrix.gammaz)
      .copy_to(&derivatives.h_gammaz[_index], tag_type());
  if constexpr (StoreJacobian) {
    Kokkos::Experimental::where(mask, jacobian_matrix.jacobian)
        .copy_to(&derivatives.h_jacobian[_index], tag_type());
  }
}

template <typename PointJacobianMatrixType,
          typename std::enable_if_t<!PointJacobianMatrixType::simd::using_simd,
                                    int> = 0>
inline void impl_store_on_host(
    const specfem::point::index<PointJacobianMatrixType::dimension_tag> &index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &derivatives,
    const PointJacobianMatrixType &jacobian_matrix) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian = PointJacobianMatrixType::store_jacobian;

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  derivatives.h_xix[_index] = jacobian_matrix.xix;
  derivatives.h_gammax[_index] = jacobian_matrix.gammax;
  derivatives.h_xiz[_index] = jacobian_matrix.xiz;
  derivatives.h_gammaz[_index] = jacobian_matrix.gammaz;
  if constexpr (StoreJacobian) {
    derivatives.h_jacobian[_index] = jacobian_matrix.jacobian;
  }
}

/**
 * @brief Load the Jacobian matrix at a given quadrature point on the device
 *
 * @ingroup ComputeJacobianMatrixDataAccess
 *
 * @tparam PointJacobianMatrixType Point Jacobian matrix type. Needs to
 * be of @ref specfem::point::jacobian_matrix
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param derivatives Jacobian matrix container
 * @param jacobian_matrix Jacobian matrix at the given quadrature point
 */
template <
    typename PointJacobianMatrixType, typename IndexType,
    typename std::enable_if_t<IndexType::using_simd ==
                                  PointJacobianMatrixType::simd::using_simd,
                              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const IndexType &index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &derivatives,
    PointJacobianMatrixType &jacobian_matrix) {
  impl_load<true>(index, derivatives, jacobian_matrix);
}

/**
 * @brief Store the Jacobian matrix at a given quadrature point on the
 * device
 *
 * @ingroup ComputeJacobianMatrixDataAccess
 *
 * @tparam PointJacobianMatrixType Point Jacobian matrix type. Needs to
 * be of @ref specfem::point::jacobian_matrix
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param derivatives Jacobian matrix container
 * @param jacobian_matrix Jacobian matrix at the given quadrature point
 */
template <
    typename PointJacobianMatrixType, typename IndexType,
    typename std::enable_if_t<IndexType::using_simd ==
                                  PointJacobianMatrixType::simd::using_simd,
                              int> = 0>
inline void load_on_host(
    const IndexType &index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &derivatives,
    PointJacobianMatrixType &jacobian_matrix) {
  impl_load<false>(index, derivatives, jacobian_matrix);
}

/**
 * @brief Store the Jacobian matrix at a given quadrature point on the
 * device
 *
 * @ingroup ComputeJacobianMatrixDataAccess
 *
 * @tparam PointJacobianMatrixType Point Jacobian matrix type. Needs to
 * be of @ref specfem::point::jacobian_matrix
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param derivatives Jacobian matrix container
 * @param jacobian_matrix Jacobian matrix at the given quadrature point
 */
template <
    typename PointJacobianMatrixType, typename IndexType,
    typename std::enable_if_t<IndexType::using_simd ==
                                  PointJacobianMatrixType::simd::using_simd,
                              int> = 0>
inline void store_on_host(
    const IndexType &index,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &derivatives,
    const PointJacobianMatrixType &jacobian_matrix) {
  impl_store_on_host(index, derivatives, jacobian_matrix);
}
} // namespace specfem::assembly
