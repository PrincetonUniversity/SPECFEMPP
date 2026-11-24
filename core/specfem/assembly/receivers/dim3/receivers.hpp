#pragma once

#include "../impl/receiver_index_base.hpp"
#include "../impl/receiver_iterator.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/receivers.hpp"

namespace specfem::assembly {

/**
 * @brief 3D assembly receivers: element location, Lagrange interpolants and
 * per-receiver rotation matrices.
 *
 * Each receiver holds a 3x3 matrix from mesh Cartesian components to the local
 * (East, North, Up) frame: the spherical frame at the station on a globe mesh,
 * the identity on a UTM/Cartesian mesh. It is applied on the device at
 * sampling time to vector seismograms. Output slots are (East, North, Up),
 * which the writer labels E/N/Z (X/Y/Z on a Cartesian mesh).
 */
template <>
struct receivers<specfem::element::dimension_tag::dim3>
    : public receivers_impl::StationIterator,
      public receivers_impl::SeismogramIterator<
          specfem::element::dimension_tag::dim3>,
      public receivers_impl::ReceiverIndexBase<
          specfem::assembly::element_types_impl::ElementSets<
              specfem::element::dimension_tag::dim3>> {

public:
  constexpr static specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag for this
                                             ///< assembly

private:
  using IndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< View to store the
                                                          ///< elements
                                                          ///< associated with
                                                          ///< the receivers
  using LagrangeInterpolantType =
      Kokkos::View<type_real ****[3], Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>; ///< View to store the
                                                   ///< Lagrange interpolant for
                                                   ///< every receiver (nrec,
                                                   ///< ngllz, nglly, ngllx, 3)

public:
  /**
   * @brief Construct a new receivers object
   *
   */
  receivers() = default;

  receivers(
      const int max_sig_step, const type_real dt, const type_real t0,
      const int nsteps_between_samples,
      const std::vector<
          std::shared_ptr<specfem::receivers::receiver<dimension_tag>>>
          &receivers,
      const std::vector<specfem::enums::wavefield> &stypes,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types);

  /**
   * @brief Get the seismogram types
   *
   * @return std::vector<specfem::enums::wavefield> Vector of seismogram types
   */
  std::vector<specfem::enums::wavefield> get_seismogram_types() const {
    return seismogram_types_;
  }

private:
  int nspec;              ///< Total number of spectral elements
  IndexViewType elements; ///< View to store the elements associated with the
                          ///< receivers
  IndexViewType::host_mirror_type h_elements; ///< Host view to store the
                                              ///< elements associated with the
                                              ///< receivers
  LagrangeInterpolantType lagrange_interpolant; ///< Lagrange interpolant for
                                                ///< every receiver
  LagrangeInterpolantType::host_mirror_type
      h_lagrange_interpolant; ///< Lagrange interpolant for every receiver
                              ///< stored on the host

  template <typename ChunkIndexType, typename ViewType>
  friend KOKKOS_FUNCTION void load_on_device(const ChunkIndexType &chunk_index,
                                             const receivers &receivers,
                                             ViewType &lagrange_interpolant);

  template <typename ChunkIndexType, typename SiesmogramViewType>
  friend KOKKOS_FUNCTION void
  store_on_device(const ChunkIndexType &chunk_index,
                  const SiesmogramViewType &seismogram_components,
                  const receivers &receivers, const bool rotate);
};

/**
 * @defgroup ComputeReceiversDataAccess3D
 * @brief 3D receiver data access functions for device computations
 *
 * These functions provide efficient access to receiver data during GPU kernel
 * execution for 3D spectral element simulations. They handle loading of
 * Lagrange interpolants and storing of seismogram components with proper
 * indexing for three-dimensional computations.
 */

/**
 * @brief Load the Lagrange interpolant for receivers associated with the
 * iterator on the device
 *
 * @ingroup ComputeReceiversDataAccess3D
 *
 * @tparam ChunkIndexType Chunk index type @ref
 * specfem::execution::ChunkElementIndex
 * @tparam ViewType Lagrange interpolant associated with the receivers in the
 * iterator
 *
 * @param chunk_index Chunk index
 * @param receivers Receivers object containing the receiver information
 * @param lagrange_interpolant Lagrange interpolant associated with the
 * receivers in the iterator
 */
template <typename ChunkIndexType, typename ViewType>
KOKKOS_FUNCTION void load_on_device(
    const ChunkIndexType &chunk_index,
    const receivers<specfem::element::dimension_tag::dim3> &receivers,
    ViewType &lagrange_interpolant) {

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;
        const int irec = index.imap;

#ifndef NDEBUG

        if (ielement >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) + ":" +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        // Load all three components for 3D
        lagrange_interpolant(ielement, index.iz, index.iy, index.ix, 0) =
            receivers.lagrange_interpolant(irec, index.iz, index.iy, index.ix,
                                           0);
        lagrange_interpolant(ielement, index.iz, index.iy, index.ix, 1) =
            receivers.lagrange_interpolant(irec, index.iz, index.iy, index.ix,
                                           1);
        lagrange_interpolant(ielement, index.iz, index.iy, index.ix, 2) =
            receivers.lagrange_interpolant(irec, index.iz, index.iy, index.ix,
                                           2);
      });

  return;
}

/**
 * @brief Store the seismogram components for receivers associated with the
 * iterator on the device
 *
 * Before you store the seismogram components, you need to set the seismogram
 * time step and type. You can do this by calling the following functions:
 * @c receivers.set_seismogram_step(isig_step);
 * @c receivers.set_seismogram_type(iseis);
 *
 * @ingroup ComputeReceiversDataAccess3D
 * @tparam ChunkIndexType Chunk index type
 * @tparam SeismogramViewType View of the seismogram components
 * @param receivers Receivers object containing the receiver information
 * @param rotate Apply the receiver's stored rotation; false for scalar
 * seismograms that occupy component 0 only
 */
template <typename ChunkIndexType, typename SeismogramViewType>
KOKKOS_FUNCTION void store_on_device(
    const ChunkIndexType &chunk_index,
    const SeismogramViewType &seismogram_components,
    const receivers<specfem::element::dimension_tag::dim3> &receivers,
    const bool rotate) {

  constexpr int ncomponents =
      specfem::element::dimension<specfem::element::dimension_tag::dim3>::dim;

  const int isig_step = receivers.get_seismogram_step();
  const int iseis = receivers.get_seis_type();

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_local_index().ispec;
        const int irec = index.imap;

#ifndef NDEBUG

        if (index.ispec >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) + ":" +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        type_real mesh_frame[ncomponents];
        for (int icomp = 0; icomp < ncomponents; ++icomp) {
          mesh_frame[icomp] = seismogram_components(ielement, icomp);
        }

        type_real output[ncomponents];
        if (rotate) {
          specfem::assembly::receivers_impl::rotate_components<ncomponents>(
              receivers.rotation_matrices, irec, mesh_frame, output);
        } else {
          for (int icomp = 0; icomp < ncomponents; ++icomp) {
            output[icomp] = mesh_frame[icomp];
          }
        }

        for (int icomp = 0; icomp < ncomponents; ++icomp) {
          receivers.seismogram_components(isig_step, iseis, irec, icomp) =
              output[icomp];
        }
      });
  return;
}

} // namespace specfem::assembly
